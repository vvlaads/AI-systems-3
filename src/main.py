import pandas as pd
import matplotlib.pyplot as plt

# === ЭТАП 1 ===
# Показывать все колонки
pd.set_option('display.max_columns', None)

# Загрузка CSV-файла
df = pd.read_csv("california_housing_train.csv")

# Вывод статистики по датасету
print(df.describe())

# Построение графиков
plt.show()

# print(df.tail(10))
# print(df.shape)
# print(df.info())

# === ЭТАП 2 ===
# Убираем пустые строки, если есть
df = df.dropna()
