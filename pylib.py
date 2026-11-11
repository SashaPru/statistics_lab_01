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

pairs_to_plot = [(0, 2), (1, 3), (0, 1), (2, 3)]
fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), layout='constrained')

for ax, (first, second) in zip(axes.flat, pairs_to_plot):
    x = data.iloc[:, first]
    y = data.iloc[:, second]
    ax.scatter(x, y, alpha=0.75)
    ax.set(xlabel=f'{x.name}, масс. %', ylabel=f'{y.name}, масс. %',
           title=f'r = {x.corr(y):.3f}')

plt.show()


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


dependence = dependence.sort_values('p', ignore_index=True)
number_of_tests = len(dependence)
multipliers = range(number_of_tests, 0, -1)
adjusted = dependence['p'] * list(multipliers)
dependence['p с поправкой Холма'] = adjusted.cummax().clip(upper=1)
dependence.round(4)


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

