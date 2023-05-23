import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
import pandas as pd
from math import sqrt
import scipy.stats as st
from scipy.stats import kstwobign
from scipy.stats import norm, t as stud_distr, chi2, uniform
np.random.seed(42)
alpha = 0.05
num_intervals = 5
sample = np.asarray([0.938, 0.531, 0.574, 0.376, 0.692, 0.764, 0.713, 0.273, 0.764, 0.939,
                     0.097, 0.872, 0.868, 0.587, 0.914, 0.582, 0.135, 0.574, 0.319, 0.991,
                     0.349, 0.722, 0.679, 0.036, 0.445, 0.743, 0.483, 0.488, 0.641, 0.324])

#sample = np.random.normal(0.5, 0.16, 30)
def draw_hist(X_n, bins = 6):
    plt.hist(X_n, bins=bins, color='steelblue', edgecolor='black')

    # Настройки легенды и заголовка
    plt.legend(['Значения'])
    plt.title('Гистограмма данных')

    # Подписи осей
    plt.xlabel('Значение')
    plt.ylabel('Частота')

    # Отображаем гистограмму
    plt.show()



def count_values_in_interval(X, interval):
    # print(interval)
    # print(np.sort(X))
    left, right = interval
    count = 0
    for i in X:
        if i <= right and i >= left:
            count += 1
    return count

# print(np.sort(sample))
# print(count_values_in_interval(sample, (0.5, 0.7)))
def generate_intervals(amount_intervals, x_1, x_n):
    interval_length = float(x_n - x_1) / amount_intervals  # вычисляем длину интервала
    intervals = [(x_1 + interval_length * i, x_1 + interval_length * (i + 1))
                 for i in range(amount_intervals)]  # создаем список интервалов
    return intervals

def probability(interval):
    a, b = interval
    prob = uniform.cdf(b, loc=0, scale=1) - uniform.cdf(a, loc=0, scale=1)
    return prob

def pearson_test(sample, amount_intervals):
    # print('sample = ', np.sort(sample))
    x_1 = min(sample)
    x_n = max(sample)
    intervals = generate_intervals(amount_intervals, x_1, x_n)
    #print(intervals)
    frequency = []
    for i in intervals:
        frequency.append(count_values_in_interval(sample, i))
        # print(i)
    stat = 0
    for i in range(len(frequency)):
        if probability(intervals[i]) == 0:
            stat = np.infty
            return stat
        stat += (((frequency[i]/sum(frequency)) - probability(intervals[i]))**2) / probability(intervals[i])
    stat = sum(frequency)*stat
    print(frequency)
    return stat
#output
distance = pearson_test(sample, num_intervals)

df = num_intervals-1
if distance < chi2.ppf(1 - alpha, df):
    print(f'Принимаем основную гипотезу с уровнем значимости {1-alpha}')
else: print('Отвергаем основную гипотезу')
p_value = 1 - chi2.cdf(distance, df)
print('Distance = ', distance)
print(f'Критическое значение для уровня значимости {1-alpha} = ', chi2.ppf(1-alpha, df))
print('p_value = ', p_value)
#statistic, p_value = st.chisquare()

draw_hist(sample,5)
#print(probability((0.2, 1)))


#print(generate_intervals(10, 0.05, 0.987))