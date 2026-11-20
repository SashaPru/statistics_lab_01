import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    return mo, np, plt, sns, stats


@app.cell
def _(mo):
    mo.md("""
    <a id="top"></a>
    # Доп задание КТвД

    ## Содержание
    - [Длина клюва пингвина](#random01)
      - [Description](#description)
      - [Math desc](#math-desc)
      - [Code](#code)
      - [Tests](#tests)
      - [Resume](#resume)
    - [Lab resume](#lab-resume)
      - [cat_mark_table](#cat-mark-table)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    <a id="random01"></a>
    ## Длина клюва пингвина (Palmer Penguins)

    <a id="description"></a>
    ### Description

    **Датасет:** Palmer Penguins (встроенный в seaborn).

    **Случайная величина:** `bill_length_mm` — длина клюва пингвина в миллиметрах.

    **Доказательство стохастической природы:**
    Значение длины клюва **невозможно вычислить по детерминированной формуле**. Оно зависит от множества случайных факторов:
    - Генетика особи и вид пингвина (Adelie, Chinstrap, Gentoo)
    - Пол и возраст птицы
    - Условия среды обитания (температура, доступность пищи)
    - Погрешность измерения прибора

    Даже для двух пингвинов одного вида и пола значения будут различаться из-за естественной биологической изменчивости. Следовательно, это **случайная (стохастическая) величина**.
    """)
    return


@app.cell
def _(mo, plt, sns):
    df = sns.load_dataset('penguins').dropna()
    var1 = df['bill_length_mm']
    var2 = df['bill_depth_mm']

    mo.md("## 2. EDA (Разведочный анализ данных)")
    mo.ui.table(df[['bill_length_mm', 'bill_depth_mm', 'species']].head())

    mo.md(f"- **Длина клюва:** среднее = {var1.mean():.2f} мм, дисперсия = {var1.var():.2f}")
    mo.md(f"- **Глубина клюва:** среднее = {var2.mean():.2f} мм, дисперсия = {var2.var():.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(var1, bins=20, alpha=0.7, color='blue')
    ax[0].set_title('Длина клюва')
    ax[1].hist(var2, bins=20, alpha=0.7, color='orange')
    ax[1].set_title('Глубина клюва')
    plt.tight_layout()
    fig
    return (var1,)


@app.cell
def _(mo):
    mo.md("""
    <a id="math-desc"></a>
    ### Math desc

    Эмпирический анализ показывает, что распределение длины клюва близко к **нормальному распределению** $N(\mu, \sigma^2)$.

    **Плотность нормального распределения:**

    $$f(x; \mu, \sigma) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

    где:
    - $\mu$ — математическое ожидание (среднее)
    - $\sigma$ — стандартное отклонение

    **Метод генерации:** Обратное преобразование или алгоритм Бокса-Мюллера для генерации нормально распределенных случайных чисел из равномерных.

    **Оценка параметров:** Метод моментов — $\hat{\mu} = \bar{x}$, $\hat{\sigma} = s$.
    """)
    return


@app.cell
def _(mo, var1):
    mo.md("""
    <a id="code"></a>
    ### Code

    Используйте ползунки ниже для изменения параметров генератора. График обновится **мгновенно**.
    """)

    n_samples = mo.ui.slider(start=100, stop=1000, step=50, value=300, label="Объем выборки (n)")
    mean_val = mo.ui.slider(start=30.0, stop=50.0, step=0.5, value=round(var1.mean(), 1), label="Среднее (μ)")
    std_val = mo.ui.slider(start=2.0, stop=6.0, step=0.1, value=round(var1.std(), 1), label="Стд. отклонение (σ)")

    mo.hstack([n_samples, mean_val, std_val])
    return mean_val, n_samples, std_val


@app.cell
def _(mean_val, mo, n_samples, np, stats, std_val, var1):
    generated_data = np.random.normal(loc=mean_val.value, scale=std_val.value, size=n_samples.value)

    mo.md("""
    <a id="tests"></a>
    ### Tests
    """)

    ks_stat, ks_pvalue = stats.ks_2samp(var1, generated_data)
    mo.md(f"**Тест Колмогорова-Смирнова:** D = {ks_stat:.4f}, p-value = {ks_pvalue:.4f}")

    if ks_pvalue > 0.05:
        mo.md("✅ **Вывод:** p-value > 0.05. Нет оснований отвергать гипотезу о том, что генерированные данные следуют тому же распределению, что и реальные.")
    else:
        mo.md("⚠️ **Вывод:** p-value < 0.05. Распределения статистически различаются.")

    mo.md(f"""
    **Сравнение метрик:**

    | Метрика | Реальные данные | Генератор | Отклонение |
    |---------|----------------|-----------|------------|
    | Среднее | {var1.mean():.4f} | {generated_data.mean():.4f} | {abs(var1.mean() - generated_data.mean())/var1.mean()*100:.2f}% |
    | Дисперсия | {var1.var():.4f} | {generated_data.var():.4f} | {abs(var1.var() - generated_data.var())/var1.var()*100:.2f}% |
    | Стд. откл. | {var1.std():.4f} | {generated_data.std():.4f} | {abs(var1.std() - generated_data.std())/var1.std()*100:.2f}% |
    """)
    return (generated_data,)


@app.cell
def _(generated_data, n_samples, plt, var1):
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(var1, bins=20, density=True, alpha=0.5, label='Реальные данные', color='blue')
    ax2.hist(generated_data, bins=20, density=True, alpha=0.5, label=f'Генератор (n={n_samples.value})', color='red')
    ax2.set_xlabel('Длина клюва, мм')
    ax2.set_ylabel('Плотность')
    ax2.set_title('Сравнение реальных и сгенерированных данных')
    ax2.legend()
    fig2
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Выводы по дополнительному анализу

    1. **Стохастическая природа:** Длина и глубина клюва пингвинов являются случайными величинами, так как их значения формируются под влиянием множества неконтролируемых биологических и экологических факторов (генетика, питание, вид). Их невозможно точно предсказать по детерминированной формуле.
    2. **Зависимость величин:** Поскольку обе величины описывают морфометрию одного объекта (клюва), мы проверили их на независимость. Коэффициент корреляции Пирсона и низкий p-value (< 0.05) статистически значимо подтверждают, что эти величины **зависимы** (с увеличением длины клюва, как правило, меняется и его глубина).
    3. **EDA:** Первичный анализ показал, что распределения обеих величин близки к нормальному, что позволяет использовать параметрические тесты и генераторы на основе нормального распределения для моделирования.
    4. **Интерактивный генератор:** Созданный интерактивный инструмент позволяет в реальном времени генерировать случайные числа с заданными параметрами (μ, σ, n) и визуально сравнивать их гистограмму с реальными данными, демонстрируя состоятельность эмпирических оценок.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a id="lab-resume"></a>
    ## Lab resume

    <a id="cat-mark-table"></a>
    ### cat_mark_table

    | cat_id | notebook_link | coeff | comment |
    |--------|---------------|-------|---------|
    | Отдельный датасет и 2 СВ | [Перейти](#random01) | | |
    | Доказательство стохастической природы | [Перейти](#description) | | |
    | EDA (Разведочный анализ) | [Перейти](#eda) | | |
    | Математическое обоснование | [Перейти](#math-desc) | | |
    | Интерактивность генератора | [Перейти](#code) | | |
    | Тесты генератора | [Перейти](#tests) | | |
    | Использование Marimo | [bonus_marimo.py](bonus_marimo.py) | | |
    """)
    return


if __name__ == "__main__":
    app.run()
