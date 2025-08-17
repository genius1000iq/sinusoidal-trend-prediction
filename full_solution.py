import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import os

# Читаем данные
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

os.makedirs("plots", exist_ok=True)
# Визуализация
plt.figure(figsize=(10, 4))
plt.plot(train['k'], train['x'], label='Train data')
plt.xlabel("k")
plt.ylabel("x")
plt.title("Данные из train.csv")
plt.legend()
plt.savefig("plots/train_data.png", dpi=300)
plt.show()


# Линейная регрессия для нахождения параметров b и c
X_lin = train[['k']]
y_lin = train['x']
lin_model = LinearRegression().fit(X_lin, y_lin)
b_init = lin_model.coef_[0]
c_init = lin_model.intercept_

print(f"Начальная оценка b = {b_init:.5f}, c = {c_init:.5f}")

# После вычитания линейной части останется синусоида и шум
residual = y_lin - (b_init * train['k'] + c_init)

plt.figure(figsize=(10, 4))
plt.plot(train['k'], residual, label='Residual (sinusoid + noise)')
plt.xlabel("k")
plt.ylabel("Residual")
plt.title("Остаток после удаления линейного тренда")
plt.legend()
plt.savefig("plots/delete_linear_part.png", dpi=300)
plt.show()


# Преобразование Фурье для вычисления частоты сигнала
fft_vals = np.fft.fft(residual)
freqs = np.fft.fftfreq(len(train['k']), d=1)  # шаг k = 1
positive_freqs = freqs[freqs > 0]
magnitudes = np.abs(fft_vals[freqs > 0])

omega_index = np.argmax(magnitudes)
freq_est = positive_freqs[omega_index]
omega_init = 2 * np.pi * freq_est

print(f"Начальная оценка omega ≈ {omega_init:.5f}")

# Можно было пойти по другому пути и использовать это разложение.
# Тогда бы мы подставлии найденную частоту
# sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
# xₖ = a * sin(omega * k + phi) + b * k + c + шум
# xₖ = a * (sin(omega * k) * cos(phi) + cos(omega * k) * sin(phi)) + b * k + c + шум
# xₖ = a * (sin(omega * k) * A + cos(omega * k) * B) + b * k + c + шум
# xₖ = A * sin(omega * k)  + B * cos(omega * k) + b * k + c + шум
# xₖ = A * f1(k)  + B * f2(k) + b * k + c + шум
# С помощью линейной регрессии можно найти A, B, b, c.
# Этот способ был бы быстрее, однако точность предсказания могла бы снизиться, по сравнению с curve_fit.

# Определение функции модели
def model_func(k, a, omega, phi, b, c):
    return a * np.sin(omega * k + phi) + b * k + c

# Нахождение точных параметров через curve_fit
p0 = [np.std(residual), omega_init, 0, b_init, c_init]  # стартовые значения
params, _ = curve_fit(model_func, train['k'], train['x'], p0=p0)

a_fit, omega_fit, phi_fit, b_fit, c_fit = params
print(f"Подобранные параметры:")
print(f"a = {a_fit:.5f}")
print(f"omega = {omega_fit:.5f}")
print(f"phi = {phi_fit:.5f}")
print(f"b = {b_fit:.5f}")
print(f"c = {c_fit:.5f}")

# 8. Оценка MSE на train
train_pred = model_func(train['k'], a_fit, omega_fit, phi_fit, b_fit, c_fit)
mse_train = mean_squared_error(train['x'], train_pred)
print(f"MSE на train: {mse_train:.8f}")

# 9. Построение предсказаний для test.csv
test_pred = model_func(test['k'], a_fit, omega_fit, phi_fit, b_fit, c_fit)
pred_df = pd.DataFrame({'k': test['k'], 'x': test_pred})

# 10. Сохранение результата
pred_df.to_csv("pred.csv", index=False)
print("Файл pred.csv сохранён.")

# 11. (Опционально) Визуализация предсказаний на train
plt.figure(figsize=(10, 4))
plt.plot(train['k'], train['x'], label='Train data', alpha=0.6)
plt.plot(train['k'], train_pred, label='Model prediction', alpha=0.8)
plt.xlabel("k")
plt.ylabel("x")
plt.title("Сравнение предсказания модели с обучающими данными")
plt.legend()
plt.savefig("plots/prediction_compare.png", dpi=300)
plt.show()
