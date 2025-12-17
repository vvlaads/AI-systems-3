import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import seaborn as sns
from config import get_resource_path


def get_coef(x_data, model):
    """Получить коэффициенты параметров для модели"""
    return pd.DataFrame({
        'Признак': x_data.columns,
        'Коэффициент': model.coef_,
        'Абс.значение': np.abs(model.coef_)
    }).sort_values('Абс.значение', ascending=False)


# Загрузка CSV
pd.set_option('display.max_columns', None)
df = pd.read_csv(get_resource_path("california_housing_train.csv"))

df = df.dropna()  # Убрать пустые строки

# СИНТЕТИЧЕСКИЙ ПРИЗНАК:
# df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']

print("=" * 100)
print("Основная статистика:")
print(df.describe())

# Построение тепловой карты
sns.heatmap(df.corr(), annot=True, fmt='.2f')
plt.tight_layout()

# Гистограммы
df.hist(bins=30,  # Количество интервалов
        figsize=(14, 8),  # Размер графика (ширина, высота)
        edgecolor="black",  # Цвет границ столбцов
        layout=(4, 3))  # Расположение графиков (строки, столбцы)
plt.suptitle('Гистограммы распределения признаков', fontsize=14)
plt.tight_layout()

# Box plot
df.plot(kind='box', subplots=True, layout=(4, 3), figsize=(14, 8),
        showfliers=False,  # Не отображать выбросы
        showmeans=True,  # Показывать среднее значение
        meanline=True,  # Отображать среднее значение линией
        meanprops={'linestyle': '--', 'linewidth': 2, 'color': 'red'})
plt.suptitle('Box-plots', fontsize=14)
plt.tight_layout()

target = "median_house_value"
X = df.drop(target, axis=1)  # сохранение входных признаков в переменную X
y = df[target]  # сохранение целевого признака в переменную y

# разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель 1: Все признаки
model1 = LinearRegression()
model1.fit(X_train, y_train)
y_predicted1 = model1.predict(X_test)
r2_1 = r2_score(y_test, y_predicted1)

# Модель 2: Число комнат
features2 = ['total_rooms', 'total_bedrooms', 'median_income']
X_train2 = X_train[features2]
X_test2 = X_test[features2]

model2 = LinearRegression()
model2.fit(X_train2, y_train)
y_predicted2 = model2.predict(X_test2)
r2_2 = r2_score(y_test, y_predicted2)

# Модель 3: Только географические признаки
features3 = ['latitude', 'longitude']
X_train3 = X_train[features3]
X_test3 = X_test[features3]

model3 = LinearRegression()
model3.fit(X_train3, y_train)
y_predicted3 = model3.predict(X_test3)
r2_3 = r2_score(y_test, y_predicted3)

# Вывод результатов
print("=" * 100)
print("Коэффициенты детерминации:")
print(f"Модель 1 (все признаки): {r2_1: .4f}")
print(f"Модель 2 (число комнат): {r2_2: .4f}")
print(f"Модель 3 (география): {r2_3: .4f}")
print("=" * 100)

print("Важность признаков в модели 1:")
print(get_coef(X, model1))
print("=" * 100)

print("Важность признаков в модели 2:")
print(get_coef(X_test2, model2))
print("=" * 100)

print("Важность признаков в модели 3:")
print(get_coef(X_test3, model3))
print("=" * 100)

plt.show()  # Отобразить графики
