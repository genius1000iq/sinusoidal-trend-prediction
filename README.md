# sinusoidal-trend-prediction

Проект по анализу и прогнозированию временного ряда с линейным трендом и синусоидальной компонентой.

## Описание
Модель аппроксимирует данные вида:
x_k = a * sin(ωk + φ) + b * k + c + ε

Реализовано:
- загрузка и предобработка данных (pandas, NumPy),
- визуализация (matplotlib),
- оценка линейного тренда (линейная регрессия),
- выделение частоты с помощью БПФ (FFT),
- подбор параметров через scipy.curve_fit,
- оценка качества модели (MSE, scikit-learn),
- сохранение предсказаний и графиков.

## Результаты
- Определены параметры модели: амплитуда, частота, фаза, коэффициенты тренда.
- Получен прогноз для тестовой выборки (pred.csv).
- Сохранены графики в папку plots.

## Запуск
git clone https://github.com/genius1000iq/sinusoidal-trend-prediction.git
cd sinusoidal-trend-prediction
pip install -r requirements.txt
python src/full_solution.py
