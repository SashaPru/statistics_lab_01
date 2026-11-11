# Химический состав образцов керамики

Статистический анализ содержания оксидов натрия (Na₂O) и алюминия
(Al₂O₃) в основе (*Body*) и глазури (*Glaze*) образцов китайской
керамики **Longquan celadon**.

В исследовании рассматриваются четыре случайные величины:

-   содержание Na₂O в основе;
-   содержание Na₂O в глазури;
-   содержание Al₂O₃ в основе;
-   содержание Al₂O₃ в глазури.

Все значения измеряются в массовых процентах. В исходном CSV-файле 88
строк: по две записи (основа и глазурь) для каждого из 44 образцов.

## Содержание

-   [Описание данных и цель
    исследования](#описание-данных-и-цель-исследования)
-   [Подготовка данных](#подготовка-данных)
-   [Исследование четырёх случайных
    величин](#исследование-четырёх-случайных-величин)
    -   [Na₂O в основе](#na₂o-в-основе)
    -   [Na₂O в глазури](#na₂o-в-глазури)
    -   [Al₂O₃ в основе](#al₂o₃-в-основе)
    -   [Al₂O₃ в глазури](#al₂o₃-в-глазури)
-   [Зависимость между величинами](#зависимость-между-величинами)
-   [Выборки и оценки](#выборки-и-оценки)
-   [Состоятельность оценок](#состоятельность-оценок)
-   [Проверка гипотезы](#проверка-гипотезы)
-   [Общие выводы](#общие-выводы)

## Описание данных и цель исследования

В лабораторной работе использован открытый датасет [Chemical Composition
of Ceramic
Samples](https://archive.ics.uci.edu/dataset/583/chemical+composition+of+ceramic+samples)
из **UC Irvine Machine Learning Repository**.

Набор данных содержит результаты энергодисперсионной рентгеновской
флуоресценции для образцов древней китайской керамики Longquan celadon.
В таблице представлены массовые доли различных оксидов в основе (*Body*)
и глазури (*Glaze*): Na₂O, MgO, Al₂O₃, SiO₂, K₂O, CaO, TiO₂, Fe₂O₃, MnO,
CuO, ZnO, PbO₂, Rb₂O, SrO, Y₂O₃, ZrO₂ и P₂O₅.

**Цель исследования:** провести статистический анализ химического
состава образцов керамики Longquan celadon, изучить распределения
компонентов в основе и глазури, а также сравнить содержание выбранных
веществ в пределах одного образца.

Измеренные значения рассматриваются как случайные величины: при выборе
другого образца содержание оксидов может отличаться.

## Подготовка данных

Для анализа строки с основой и глазурью объединяются по идентификатору
образца. После объединения формируется таблица из четырёх столбцов с
исследуемыми величинами.

**Используемые библиотеки:** NumPy, pandas, Matplotlib и SciPy.

```{=html}
<details>
```
```{=html}
<summary>
```
Импорт библиотек и загрузка данных
```{=html}
</summary>
```
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import display
from itertools import combinations

df = pd.read_csv('Chemical Composion of Ceramic.csv')
body = df[df['Part'] == 'Body'].copy()
glaze = df[df['Part'] == 'Glaze'].copy()

body['Образец'] = body['Ceramic Name'].str.removesuffix('-b')
glaze['Образец'] = glaze['Ceramic Name'].str.removesuffix('-g')

paired = body.merge(glaze, on='Образец', suffixes=('_body', '_glaze'), validate='one_to_one')
assert len(paired) == len(body) == len(glaze)

data = paired.set_index('Образец')[['Na2O_body', 'Na2O_glaze', 'Al2O3_body', 'Al2O3_glaze']]
data.columns = ['Na₂O в основе', 'Na₂O в глазури', 'Al₂O₃ в основе', 'Al₂O₃ в глазури']
assert data.notna().all().all()
print(f'Строк в CSV: {len(df)}. Полных пар: {len(data)}. Пропусков в этих четырёх столбцах нет.')
data.head()
```


    </details>

    После загрузки проверяется количество строк, число полных пар образцов и отсутствие пропусков в четырёх выбранных столбцах.

    ## Исследование четырёх случайных величин

    Для каждой величины рассчитываются основные описательные статистики: среднее, стандартное отклонение, минимум, квартили, медиана, максимум, дисперсия и асимметрия.

    Также строятся:
    - гистограмма;
    - диаграмма размаха;
    - гистограмма стандартизированных значений;
    - Q–Q график для сравнения с нормальной моделью.

    При стандартизации используется формула \(z=(x-\bar{x})/s\). Для проверки нормальности применяется критерий Шапиро-Уилка. При \(p<0{,}05\) нормальная модель отвергается.


    <details>
    <summary>Функция для описательной статистики и построения графиков</summary>

    ```python
    def describe_and_plot(values):
        summary = values.describe().rename(index={
            'count': 'Число значений', 'mean': 'Среднее', 'std': 'Стандартное отклонение',
            'min': 'Минимум', '25%': 'Первый квартиль', '50%': 'Медиана',
            '75%': 'Третий квартиль', 'max': 'Максимум',
        })
        summary['Дисперсия'] = values.var()
        summary['Асимметрия'] = values.skew()
        display(summary.round(3).to_frame(values.name))

        z = (values - values.mean()) / values.std()
        test = stats.shapiro(values)
        print(f'После стандартизации: среднее {z.mean():.2f}, стандартное отклонение {z.std():.2f}.')
        print(f'Шапиро — Уилк: W = {test.statistic:.4f}, p = {test.pvalue:.4f}.')

        fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), layout='constrained')
        hist, box, scaled, qq = axes.flat
        hist.hist(values, bins=10, edgecolor='white')
        hist.set(title='Гистограмма', xlabel='Доля, масс. %', ylabel='Число образцов')
        box.boxplot(values)
        box.set(title='Диаграмма размаха', ylabel='Доля, масс. %', xticks=[])
        scaled.hist(z, bins=10, edgecolor='white')
        scaled.set(title='После стандартизации', xlabel='z', ylabel='Число образцов')
        stats.probplot(values, dist='norm', plot=qq)
        qq.set(title='Q–Q график', xlabel='Квантили нормальной модели', ylabel='Доля, масс. %')
        fig.suptitle(values.name)
        plt.show()

```{=html}
</details>
```
### Na₂O в основе

Доля оксида натрия в основе выбранного образца. Единица измерения —
масс. %.

```{=html}
<details>
```
```{=html}
<summary>
```
Расчёт статистик и построение графиков для Na₂O в основе
```{=html}
</summary>
```
``` python
describe_and_plot(data['Na₂O в основе'])
```

```{=html}
</details>
```
Среднее значение составляет 0,398 масс. %, медиана — 0,370 масс. %. По
тесту Шапиро-Уилка $p=0,3839$ нет оснований отвергать нормальную
модель.

### Na₂O в глазури

Доля оксида натрия в глазури выбранного образца. Единица измерения —
масс. %.

```{=html}
<details>
```
```{=html}
<summary>
```
Расчёт статистик и построение графиков для Na₂O в глазури
```{=html}
</summary>
```
``` python
describe_and_plot(data['Na₂O в глазури'])
```

```{=html}
</details>
```
Среднее значение составляет 0,546 масс. %, медиана — 0,375 масс. %. На
гистограмме заметен правый хвост. По тесту Шапиро-Уилка нормальная
модель отвергается, поэтому дополнительно рассматриваются логнормальная
и гамма-модели.

Для логнормальной модели проверяется нормальность логарифмов значений.
Для гамма-распределения параметры подбираются по данным, а качество
модели оценивается с помощью параметрического бутстрэпа и критерия
Колмогорова-Смирнова.

```{=html}
<details>
```
```{=html}
<summary>
```
Проверка логнормальной и гамма-модели для Na₂O в глазури
```{=html}
</summary>
```
``` python
sodium_glaze = data['Na₂O в глазури'].to_numpy()
log_test = stats.shapiro(np.log(sodium_glaze))
print(f'Логнормальная модель: p = {log_test.pvalue:.4f}')

shape, location, scale = stats.gamma.fit(sodium_glaze, floc=0)
gamma_test = stats.goodness_of_fit(
    stats.gamma, sodium_glaze, known_params={'loc': 0},
    statistic='ks', n_mc_samples=1999, rng=42,
)
print(f'Гамма-модель: k = {shape:.4f}, θ = {scale:.4f}')
print(f'D = {gamma_test.statistic:.4f}, p = {gamma_test.pvalue:.4f}')

x = np.linspace(0, sodium_glaze.max() + 0.3, 200)
plt.hist(sodium_glaze, bins=10, density=True, edgecolor='white', label='Данные')
plt.plot(x, stats.gamma.pdf(x, shape, loc=location, scale=scale), label='Гамма-модель')
plt.xlabel('Доля Na₂O в глазури, масс. %')
plt.ylabel('Плотность')
plt.legend()
plt.show()
```

```{=html}
</details>
```
Для логнормальной модели получено $p=0,0137$, поэтому она отвергается.
Для гамма-модели получено $p=0,7605$. По результатам теста
гамма-распределение можно использовать как приближение.

### Al₂O₃ в основе

Доля оксида алюминия в основе выбранного образца. Единица измерения —
масс. %.

```{=html}
<details>
```
```{=html}
<summary>
```
Расчёт статистик и построение графиков для Al₂O₃ в основе
```{=html}
</summary>
```
``` python
describe_and_plot(data['Al₂O₃ в основе'])
```

```{=html}
</details>
```
Среднее и медиана почти совпадают: 21,812 и 21,835 масс. %
соответственно. При $p=0,400$ нет оснований отвергать нормальную
модель.

### Al₂O₃ в глазури

Доля оксида алюминия в глазури выбранного образца. Единица измерения —
масс. %.

```{=html}
<details>
```
```{=html}
<summary>
```
Расчёт статистик и построение графиков для Al₂O₃ в глазури
```{=html}
</summary>
```
``` python
describe_and_plot(data['Al₂O₃ в глазури'])
```

```{=html}
</details>
```
Среднее значение составляет 13,110 масс. %. Разброс значений в глазури
меньше, чем в основе: стандартные отклонения равны 0,849 и 2,301
процентного пункта соответственно. При $p=0,359$ нет оснований
отвергать нормальную модель.

## Зависимость между величинами

Сначала строятся диаграммы рассеяния для четырёх пар величин. Каждая
точка соответствует одному образцу. Коэффициент корреляции Пирсона (r)
характеризует линейную связь.

```{=html}
<details>
```
```{=html}
<summary>
```
Диаграммы рассеяния и коэффициенты корреляции Пирсона
```{=html}
</summary>
```
``` python
pairs_to_plot = [(0, 2), (1, 3), (0, 1), (2, 3)]
fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), layout='constrained')

for ax, (first, second) in zip(axes.flat, pairs_to_plot):
    x = data.iloc[:, first]
    y = data.iloc[:, second]
    ax.scatter(x, y, alpha=0.75)
    ax.set(xlabel=f'{x.name}, масс. %', ylabel=f'{y.name}, масс. %',
           title=f'r = {x.corr(y):.3f}')

plt.show()
```

```{=html}
</details>
```
Затем проверяется независимость всех шести пар. В качестве меры связи
используется квадрат дистанционной ковариации, который позволяет
обнаруживать не только линейные зависимости. Значимость оценивается
перестановочным тестом: значения одной величины случайным образом
переставляются между образцами.

```{=html}
<details>
```
```{=html}
<summary>
```
Вычисление дистанционной ковариации и перестановочный тест
```{=html}
</summary>
```
``` python
def centered_distances(values):
    distances = np.abs(values[:, None] - values[None, :])
    row_means = distances.mean(axis=1, keepdims=True)
    column_means = distances.mean(axis=0)
    return distances - row_means - column_means + distances.mean()


rng = np.random.default_rng(301)
orders = [rng.permutation(len(data)) for _ in range(4999)]
results = []

for first, second in combinations(data.columns, 2):
    a = centered_distances(data[first].to_numpy())
    b = centered_distances(data[second].to_numpy())
    observed = (a * b).mean()

    at_least_as_large = 0
    for order in orders:
        shuffled = b[np.ix_(order, order)]
        if (a * shuffled).mean() >= observed - 1e-12:
            at_least_as_large += 1

    p = (at_least_as_large + 1) / (len(orders) + 1)
    results.append([first, second, data[first].corr(data[second]), p])

dependence = pd.DataFrame(results, columns=['Величина 1', 'Величина 2', 'r Пирсона', 'p'])
```

```{=html}
</details>
```
Поскольку выполняются шесть проверок, для p-значений применяется
поправка Холма. Скорректированные значения сравниваются с уровнем
значимости 0,05.

```{=html}
<details>
```
```{=html}
<summary>
```
Поправка Холма для множественных проверок
```{=html}
</summary>
```
``` python
dependence = dependence.sort_values('p', ignore_index=True)
number_of_tests = len(dependence)
multipliers = range(number_of_tests, 0, -1)
adjusted = dependence['p'] * list(multipliers)
dependence['p с поправкой Холма'] = adjusted.cummax().clip(upper=1)
dependence.round(4)
```

```{=html}
</details>
```
Независимость отвергается для двух пар: Na₂O в основе и глазури
$(p=0,018)$, а также Na₂O и Al₂O₃ внутри основы $(p=0,032)$. Для
остальных пар данных недостаточно, чтобы сделать вывод о зависимости.
Перестановочный тест предполагает возможность свободно перемешивать
строки при независимости; различия между группами образцов также могут
влиять на результат.

## Выборки и оценки

Проверяется, насколько небольшая выборка описывает все 44 образца. Для
каждой схемы отбора выбираются 20 идентификаторов, причём все четыре
измерения одного образца берутся вместе.

-   **С возвращением:** один и тот же образец может попасть в выборку
    несколько раз.
-   **Без возвращения:** каждый образец может попасть в выборку не более
    одного раза.

Среднее и дисперсия выборки сравниваются со значениями по полному набору
данных. Для дисперсии полного набора используется делитель $N=44$, а для
выборочной дисперсии — $n-1$. При отборе без возвращения выборочная
дисперсия дополнительно умножается на $\frac{N-1}{N}$.

```{=html}
<details>
```
```{=html}
<summary>
```
Сравнение оценок по выборкам с возвращением и без возвращения
```{=html}
</summary>
```
``` python
N = len(data)
full_mean = data.mean()
full_variance = data.var(ddof=0)
methods = {True: 'С возвращением', False: 'Без возвращения'}
rng = np.random.default_rng(91)

comparison = pd.DataFrame({'Среднее по CSV': full_mean, 'Дисперсия по CSV': full_variance})

for replace, method in methods.items():
    indices = rng.choice(N, size=20, replace=replace)
    sample = data.iloc[indices]
    variance = sample.var(ddof=1)
    if not replace:
        variance *= (N - 1) / N

    print(f'{method}: {sample.index.nunique()} разных образцов из 20.')
    comparison[f'Среднее · {method.lower()}'] = sample.mean()
    comparison[f'Дисперсия · {method.lower()}'] = variance

display(comparison.filter(like='Среднее').round(4))
display(comparison.filter(like='Дисперсия').round(4))
```

```{=html}
</details>
```
## Состоятельность оценок

Состоятельность означает, что при увеличении размера выборки оценка с
большой вероятностью приближается к истинному значению. Для каждого
размера выборки выполняется 2 000 повторов.

В качестве меры ошибки используется RMSE — корень из среднего квадрата
отклонений оценки от значения, рассчитанного по всему набору данных. Чем
ниже значение RMSE, тем точнее оценка.

```{=html}
<details>
```
```{=html}
<summary>
```
Моделирование выборок и расчёт RMSE
```{=html}
</summary>
```
``` python
def rmse(estimates, target):
    return np.sqrt(((estimates - target) ** 2).mean(axis=0))


rng = np.random.default_rng(2026)
sizes = [5, 10, 20, 30, N]
repeats = 2000
values = data.to_numpy()
errors = {}

for replace, method in methods.items():
    mean_errors = []
    variance_errors = []
    for n in sizes:
        if replace:
            indices = rng.integers(N, size=(repeats, n))
        else:
            indices = np.array([rng.choice(N, size=n, replace=False) for _ in range(repeats)])

        samples = values[indices]
        sample_means = samples.mean(axis=1)
        sample_variances = samples.var(axis=1, ddof=1)
        if not replace:
            sample_variances *= (N - 1) / N

        mean_errors.append(rmse(sample_means, full_mean.to_numpy()))
        variance_errors.append(rmse(sample_variances, full_variance.to_numpy()))

    errors[method] = np.array(mean_errors), np.array(variance_errors)

assert np.allclose(errors['Без возвращения'][0][-1], 0)
assert np.allclose(errors['Без возвращения'][1][-1], 0)
```

```{=html}
</details>
```
```{=html}
<details>
```
```{=html}
<summary>
```
Графики зависимости ошибки от размера выборки
```{=html}
</summary>
```
``` python
fig, axes = plt.subplots(4, 2, figsize=(8, 10), layout='constrained', sharex=True)

for i, name in enumerate(data.columns):
    for method, (mean_errors, variance_errors) in errors.items():
        axes[i, 0].plot(sizes, mean_errors[:, i], 'o-', label=method)
        axes[i, 1].plot(sizes, variance_errors[:, i], 'o-', label=method)
    axes[i, 0].set(title=f'{name} · среднее', ylabel='RMSE, п.п.')
    axes[i, 1].set(title=f'{name} · дисперсия', ylabel='RMSE, (п.п.)²')

axes[0, 0].legend(fontsize=8)
for ax in axes[-1]:
    ax.set(xlabel='Размер выборки', xticks=sizes)

plt.show()
```

```{=html}
</details>
```
На графиках видно, что ошибки уменьшаются с ростом выборки. При отборе
без возвращения среднее обычно оценивается точнее при том же размере
выборки. Для среднего теоретическое отношение дисперсий оценок при
отборе без возвращения и с возвращением равно $\frac{N-n}{N-1}$. При $n=20$
в эксперименте RMSE среднего составил примерно 71-75 % от ошибки при
отборе с возвращением.

## Проверка гипотезы

**Исследовательский вопрос:** отличается ли средняя доля Na₂O в глазури
и в основе?

-   $H_0$: средняя разность «глазурь минус основа» равна нулю.
-   $H_1$: средняя разность отличается от нуля.

Так как измерения получены для одних и тех же образцов, используется
парный t-тест. Перед его применением проверяется нормальность разностей.
Уровень значимости — 0,05.

```{=html}
<details>
```
```{=html}
<summary>
```
Проверка нормальности разностей и парный t-тест
```{=html}
</summary>
```
``` python
sodium_body = data['Na₂O в основе']
sodium_glaze = data['Na₂O в глазури']
difference = sodium_glaze - sodium_body

normality = stats.shapiro(difference)
test = stats.ttest_rel(sodium_glaze, sodium_body)
interval = test.confidence_interval(confidence_level=0.95)

print(f'Шапиро — Уилк для разностей: p = {normality.pvalue:.4f}')
print(f'Средняя разность: {difference.mean():.3f} п.п.')
print(f'Парный t-тест: t = {test.statistic:.3f}, p = {test.pvalue:.4f}')
print(f'95%-й доверительный интервал: от {interval.low:.3f} до {interval.high:.3f} п.п.')
```

```{=html}
</details>
```
Для разностей тест Шапиро-Уилка даёт $p=0,276$, поэтому
нормальность не отвергается. Парный t-тест даёт $p=0,0205$,
следовательно, нулевая гипотеза отвергается на уровне значимости 0,05. В
рассмотренных данных доля Na₂O в глазури в среднем выше на 0,148
процентного пункта.

## Общие выводы

1.  Для Na₂O в основе и Al₂O₃ в обеих частях нормальная модель может
    использоваться как приближение. Для Na₂O в глазури подошла
    гамма-модель.
2.  Статистическая связь обнаружена между содержанием Na₂O в основе и
    глазури, а также между Na₂O и Al₂O₃ внутри основы.
3.  С увеличением размера выборки оценки среднего и дисперсии становятся
    точнее. При отборе без возвращения среднее обычно оценивается точнее
    при том же размере выборки.
4.  По 44 парам образцов доля Na₂O в глазури в среднем оказалась выше,
    чем в основе. Образцы были заранее отобраны составителями набора,
    поэтому результат нельзя безоговорочно распространять на всю
    историческую керамику.

## Запуск проекта

1.  Скачайте CSV-файл с [UC Irvine Machine Learning
    Repository](https://archive.ics.uci.edu/dataset/583/chemical+composition+of+ceramic+samples).
2.  Поместите CSV-файл в ту же папку, что и ноутбук.
3.  Установите необходимые библиотеки:

``` bash
pip install numpy pandas matplotlib scipy
```

4.  Откройте ноутбук в Jupyter Notebook или Google Colab и выполните
    ячейки по порядку.

**Важно:** в коде используется имя файла
`Chemical Composion of Ceramic.csv` (именно в таком написании оно
указано в ноутбуке). Если скачанный файл называется иначе, измените имя
в строке `pd.read_csv(...)`.
