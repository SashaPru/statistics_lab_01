import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy import special
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

def plot_pairs(data, pairs_to_plot):
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

def test_independence(data, n_permutations=4999, seed=301):
    from itertools import combinations
    
    rng = np.random.default_rng(seed)
    orders = [rng.permutation(len(data)) for _ in range(n_permutations)]
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
    
    return pd.DataFrame(results, columns=['Величина 1', 'Величина 2', 'r Пирсона', 'p'])


def holm_correction(dependence):
    dependence = dependence.sort_values('p', ignore_index=True)
    number_of_tests = len(dependence)
    multipliers = range(number_of_tests, 0, -1)
    adjusted = dependence['p'] * list(multipliers)
    dependence['p с поправкой Холма'] = adjusted.cummax().clip(upper=1)
    return dependence.round(4)


def compare_sampling(data, sample_size=20, seed=91):
    N = len(data)
    full_mean = data.mean()
    full_variance = data.var(ddof=0)
    methods = {True: 'С возвращением', False: 'Без возвращения'}
    rng = np.random.default_rng(seed)
    
    comparison = pd.DataFrame({'Среднее по CSV': full_mean, 'Дисперсия по CSV': full_variance})
    
    for replace, method in methods.items():
        indices = rng.choice(N, size=sample_size, replace=replace)
        sample = data.iloc[indices]
        variance = sample.var(ddof=1)
        if not replace:
            variance *= (N - 1) / N
        
        print(f'{method}: {sample.index.nunique()} разных образцов из {sample_size}.')
        comparison[f'Среднее · {method.lower()}'] = sample.mean()
        comparison[f'Дисперсия · {method.lower()}'] = variance
    
    display(comparison.filter(like='Среднее').round(4))
    display(comparison.filter(like='Дисперсия').round(4))
    
    return comparison


def compute_rmse_errors(data, sizes=None, repeats=2000, seed=2026):
    N = len(data)
    full_mean = data.mean()
    full_variance = data.var(ddof=0)
    methods = {True: 'С возвращением', False: 'Без возвращения'}
    
    if sizes is None:
        sizes = [5, 10, 20, 30, N]
    
    rng = np.random.default_rng(seed)
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
            
            mean_errors.append(np.sqrt(((sample_means - full_mean.to_numpy()) ** 2).mean(axis=0)))
            variance_errors.append(np.sqrt(((sample_variances - full_variance.to_numpy()) ** 2).mean(axis=0)))
        
        errors[method] = np.array(mean_errors), np.array(variance_errors)
    
    assert np.allclose(errors['Без возвращения'][0][-1], 0)
    assert np.allclose(errors['Без возвращения'][1][-1], 0)
    
    return errors, sizes


def plot_rmse_errors(data, errors, sizes):
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


def paired_ttest_analysis(data, col1='Na2O в основе', col2='Na2O в глазури', confidence_level=0.95):
    series1 = data[col1]
    series2 = data[col2]
    difference = series2 - series1
    
    normality = stats.shapiro(difference)
    test = stats.ttest_rel(series2, series1)
    interval = test.confidence_interval(confidence_level=confidence_level)
    
    print(f'Шапиро-Уилк для разностей: p = {normality.pvalue:.4f}')
    print(f'Средняя разность: {difference.mean():.3f} п.п.')
    print(f'Парный t-тест: t = {test.statistic:.3f}, p = {test.pvalue:.4f}')
    print(f'{int(confidence_level*100)}%-й доверительный интервал: от {interval.low:.3f} до {interval.high:.3f} п.п.')
    
    return {
        'shapiro_p': normality.pvalue,
        'mean_diff': difference.mean(),
        't_statistic': test.statistic,
        'p_value': test.pvalue,
        'ci_low': interval.low,
        'ci_high': interval.high
    }


real_na_glaze = data['Na₂O в глазури'].to_numpy(dtype=float)
k_hat, loc_hat, theta_hat = stats.gamma.fit(real_na_glaze, floc=0)
assert loc_hat == 0 and k_hat > 0 and theta_hat > 0

print(f'n = {len(real_na_glaze)}; k = {k_hat:.6f}; θ = {theta_hat:.6f} масс. %')
print(f'Теория: E[X] = {k_hat * theta_hat:.4f} масс. %; '
      f'Var[X] = {k_hat * theta_hat**2:.4f} (масс. %)^2')


def normal_box_muller(size, rng):
    """Стандартные нормальные значения из независимых равномерных чисел."""
    u1 = 1.0 - rng.random(size)
    u2 = rng.random(size)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

def gamma_marsaglia_tsang(k, theta, size, rng, diagnostics=False):
    """Независимые Gamma(k, theta) методом принятия и отклонения.

    Parameters
    ----------
    k : float                 Параметр формы, строго положительный.
    theta : float             Масштаб, строго положительный.
    size : int                Число значений, неотрицательное целое.
    rng : numpy.random.Generator
    diagnostics : bool        Если True, вернуть также число кандидатов.
    """
    if not np.isfinite(k) or k <= 0 or not np.isfinite(theta) or theta <= 0:
        raise ValueError('k и theta должны быть конечными положительными числами')
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 0:
        raise ValueError('size должен быть неотрицательным целым числом')
    if not isinstance(rng, np.random.Generator):
        raise TypeError('rng должен быть numpy.random.Generator')

    effective_k = k if k >= 1 else k + 1
    d = effective_k - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)
    result = np.empty(size, dtype=float)
    pending = np.arange(size)
    proposals = 0

    while pending.size:
        z = normal_box_muller(pending.size, rng)
        u = 1.0 - rng.random(pending.size)
        w = 1.0 + c * z
        valid = w > 0
        # Для отклонённых w<=0 подставляем v=1, чтобы не вычислять log(v<=0).
        v = np.where(valid, w**3, 1.0)
        log_ratio = 0.5 * z**2 + d * (1.0 - v + np.log(v))
        accepted = valid & (np.log(u) < log_ratio)
        result[pending[accepted]] = d * v[accepted]
        proposals += pending.size
        pending = pending[~accepted]

    if k < 1:
        result *= (1.0 - rng.random(size)) ** (1.0 / k)
    result *= theta
    return (result, proposals) if diagnostics else result


generated, proposals = gamma_marsaglia_tsang(
    k_hat, theta_hat, 50_000, np.random.default_rng(20261111), diagnostics=True
)
print(f'Получено {len(generated):,} значений из {proposals:,} кандидатов; '
      f'доля принятых = {len(generated) / proposals:.3%}')
print('Первые десять значений, масс. %:', np.round(generated[:10], 3))


seed = 7341
a = gamma_marsaglia_tsang(k_hat, theta_hat, 1000, np.random.default_rng(seed))
b = gamma_marsaglia_tsang(k_hat, theta_hat, 1000, np.random.default_rng(seed))
assert np.array_equal(a, b)
assert len(a) == 1000 and np.all(np.isfinite(a)) and np.all(a > 0)
assert gamma_marsaglia_tsang(1, 1, 0, np.random.default_rng(seed)).size == 0

for bad in [(0, 1, 10), (1, 0, 10), (1, 1, -1), (1, 1, 2.5)]:
    try:
        gamma_marsaglia_tsang(*bad, rng=np.random.default_rng(seed))
    except ValueError:
        pass
    else:
        raise AssertionError(f'Аргументы {bad} должны вызывать ValueError')

test_rows = []
for test_k, test_theta in [(0.6, 0.8), (1.0, 0.8), (k_hat, theta_hat), (4.0, 0.8)]:
    draws = gamma_marsaglia_tsang(test_k, test_theta, 40_000,
                                  np.random.default_rng(7300 + round(100 * test_k)))
    theory_mean = test_k * test_theta
    theory_var = test_k * test_theta**2
    D, p = stats.kstest(draws, 'gamma', args=(test_k, 0, test_theta))
    mean_error_se = abs(draws.mean() - theory_mean) / np.sqrt(theory_var / len(draws))
    var_relative_error = abs(draws.var(ddof=1) / theory_var - 1)
    lag1 = np.corrcoef(draws[:-1], draws[1:])[0, 1]
    assert mean_error_se < 5
    assert var_relative_error < 0.08
    assert D < 2 / np.sqrt(len(draws))
    assert abs(lag1) < 0.04
    test_rows.append([test_k, test_theta, mean_error_se, var_relative_error, D, p, lag1])

tests = pd.DataFrame(test_rows, columns=[
    'k', 'θ', '|ошибка среднего| / SE', 'Отн. ошибка дисперсии', 'KS D', 'KS p', 'r соседних'
])
display(tests.round(4))
print('Все автопроверки пройдены.')


def sample_metrics(x, axis=None):
    return {
        'Среднее, масс. %': np.mean(x, axis=axis),
        'Дисперсия, (масс. %)^2': np.var(x, axis=axis, ddof=1),
        'Медиана, масс. %': np.median(x, axis=axis),
        'Q10, масс. %': np.quantile(x, 0.10, axis=axis),
        'Q90, масс. %': np.quantile(x, 0.90, axis=axis),
    }

n_real = len(real_na_glaze)
replications = 5000
replicated = gamma_marsaglia_tsang(
    k_hat, theta_hat, replications * n_real, np.random.default_rng(4401)
).reshape(replications, n_real)
real_metrics = sample_metrics(real_na_glaze)
generator_metrics = sample_metrics(generated)
replicated_metrics = sample_metrics(replicated, axis=1)

comparison_rows = []
for name, observed in real_metrics.items():
    lower, upper = np.quantile(replicated_metrics[name], [0.025, 0.975])
    comparison_rows.append([
        name, observed, generator_metrics[name], lower, upper,
        'да' if lower <= observed <= upper else 'нет'
    ])
comparison_s2 = pd.DataFrame(comparison_rows, columns=[
    'Метрика', 'CSV (n=44)', 'Генератор (n=50 000)',
    'Нижняя граница 95%', 'Верхняя граница 95%', 'CSV внутри?'
]).set_index('Метрика')
display(comparison_s2.round(4))

ks_generator = stats.kstest(generated, 'gamma', args=(k_hat, 0, theta_hat))
print(f'KS для генератора против заданной Gamma(k, θ): '
      f'D={ks_generator.statistic:.4f}, p={ks_generator.pvalue:.4f}')


grid = np.linspace(0, max(real_na_glaze.max(), np.quantile(generated, 0.995)) * 1.06, 350)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')

axes[0].hist(real_na_glaze, bins=10, density=True, alpha=0.55,
             edgecolor='white', label='CSV: 44 образца')
axes[0].hist(generated, bins=55, density=True, alpha=0.32,
             edgecolor='none', label='Генератор: 50 000')
axes[0].plot(grid, stats.gamma.pdf(grid, k_hat, scale=theta_hat),
             color='black', lw=1.8, label='Плотность модели')
axes[0].set(xlabel='Na₂O в глазури, масс. %', ylabel='Плотность',
            title='Форма распределения', xlim=(0, grid.max()))
axes[0].legend(fontsize=8)

real_sorted = np.sort(real_na_glaze)
sim_sorted = np.sort(generated)
axes[1].step(real_sorted, np.arange(1, n_real + 1) / n_real,
             where='post', label='Эмпирическая F: CSV')
axes[1].step(sim_sorted, np.arange(1, len(generated) + 1) / len(generated),
             where='post', alpha=0.75, label='Эмпирическая F: генератор')
axes[1].plot(grid, stats.gamma.cdf(grid, k_hat, scale=theta_hat),
             color='black', lw=1.8, label='F модели')
axes[1].set(xlabel='Na₂O в глазури, масс. %', ylabel='F(x)',
            title='Функция распределения', xlim=(0, grid.max()), ylim=(0, 1.02))
axes[1].legend(fontsize=8)
plt.show()

def fit_gamma_distribution(data, column_name):
    """Подбирает параметры гамма-расделения для заданного столбца."""
    values = data[column_name].to_numpy(dtype=float)
    k_hat, loc_hat, theta_hat = stats.gamma.fit(values, floc=0)
    
    # Проверка корректности подбора
    assert loc_hat == 0 and k_hat > 0 and theta_hat > 0, "Ошибка подбора параметров гамма-распределения"
    
    return k_hat, loc_hat, theta_hat