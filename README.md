# Химический состав образцов керамики

Сравним, сколько Na₂O и Al₂O₃ содержится в основе (Body) и глазури (Glaze). Оба вещества измерены в массовых процентах. В CSV 88 строк, по две на каждый из 44 образцов.

[Четыре величины](#rv01) · [Зависимость](#dependence) · [Выборки и оценки](#sampling) · [Гипотеза](#hypothesis) · [Выводы](#resume)

---

## Описание датасета

В лабораторной работе использовался открытый датасет UC Irvine Machine Learning Repository: https://archive.ics.uci.edu/dataset/583/chemical+composition+of+ceramic+samples

Открытый репозиторий представляет нам классификацию образцов древней китайской керамики - Longquan celadon на основе их химического состава на основе энергодисперсионной рентгеновской флуоресценции. Для изучения сырья и технологии обжига они отобрали 40 типичных осколков, относящихся к четырем культурным эпохам.

В таблице представлены процентные содержания таких соединений как: Na2O, MgO, Al2O3, SiO2, K2O, CaO, TiO2, Fe2O3, MnO, CuO, ZnO, PbO2, Rb2O, SrO, Y2O3, ZrO2, P2O5. Их содержание представлено в основах(body) и в глазури(glaze) остатков.

**Цель исследования:** провести статистический анализ химического состава образцов керамики Longquan celadon; исследовать особенности распределения компонентов в body и glaze; сравнить процентное содержание веществ в body и glaze в рамках 1 остатка керамики.

В рамках данной лабораторной работы мы будем подробно изучать 4 случайные величины: процентное содержание Na2O в body, процентное содержание Na2O в glaze, процентное содержание Al2O3 в body, процентное содержание Al2O3 в glaze.

---

## Почему рассматриваем данные как случайные величины

Когда выбираем один образец, у него есть четыре числа, заранее эти числа нельзя узнать. В исходном исследовании значения определили методом рентгенофлуоресцентного анализа. При выборе другого образца значения могут поменяться. Поэтому рассматриваем их как случайные величины.

---

<details>
<summary>📦 <b>Код: Импорт библиотек и загрузка данных</b> (нажми, чтобы развернуть)</summary>

```python
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
