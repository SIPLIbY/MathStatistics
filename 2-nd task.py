import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
import pandas as pd
from math import sqrt
import scipy.stats as st
from scipy.stats import kstwobign, kstest

from scipy.stats import norm as norm_distr, t as stud_distr, chi2 as hi2_distr
np.random.seed(42)
bins = 5
alpha = 0.05
sample = np.asarray([0.938, 0.531, 0.574, 0.376, 0.692, 0.764, 0.713, 0.273, 0.764, 0.939,
                     0.097, 0.872, 0.868, 0.587, 0.914, 0.582, 0.135, 0.574, 0.319, 0.991,
                     0.349, 0.722, 0.679, 0.036, 0.445, 0.743, 0.483, 0.488, 0.641, 0.324])


#sample = np.random.uniform(0, 1, 1000)
#sample = np.random.normal(0, 1, 15)
def draw_hist(X_n):
    plt.hist(X_n, bins=bins, color='steelblue', edgecolor='black')

    # Настройки легенды и заголовка
    plt.legend(['Значения'])
    plt.title('Гистограмма данных')

    # Подписи осей
    plt.xlabel('Значение')
    plt.ylabel('Частота')

    # Отображаем гистограмму
    plt.show()

def draw_graphic():
    plt.figure(figsize=(9, 9))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.hlines(0, -1, sample_sorted[0], linewidth=1.5, color='black')
    plt.hlines(1, sample_sorted[n - 1], 2, linewidth=1.5, color='black')
    for j in range(n - 1): plt.hlines(ecdf[j + 1], sample_sorted[j], sample_sorted[j + 1], linewidth=1.5, color='black')
    plt.plot([-1, 0, 1, 2], [0, 0, 1, 1], linewidth=1.5, color='purple')
    plt.grid(True)
    plt.show()

sample_sorted = np.sort(sample)
n = sample.size

ecdf = np.where(sample_sorted <= sample_sorted)[0] / n
ecdf_sm = sm.distributions.ECDF(sample)
print(ecdf.size)
D_exactly = max(max(abs(ecdf - sample_sorted)), abs(sample_sorted[0]), abs(1-sample_sorted[n-1]))

x_space = np.linspace(min(sample), max(sample), num=1000000)



cdf_unifrom = st.uniform.cdf(x_space)
df = pd.DataFrame({'x': x_space, 'ecdf': ecdf_sm(x_space), 'cdf_norm': cdf_unifrom})
max_diff = np.max(np.abs(df['ecdf'] - df['cdf_norm']))

print('sqrt*D = ', max_diff*sqrt(n))
print("D = ", max_diff)
quantile = kstwobign.ppf(1-alpha)
print('Критическое значение = ', quantile)
print("quant = ", quantile)
if max_diff*sqrt(n) < quantile:
    print(f'Гипотеза принимается с уровнем значимости {1-alpha}')
else: print('Гипотеза отвергается')
p_value = 1 - kstwobign.cdf(sqrt(n)*max_diff)
#p_value = kstwobign.cdf(0.684653197)
#print("Проверка", 1 - kstwobign.cdf(sqrt(n)*0.18566666666666667))
print("p_value = ", p_value)
#draw_hist(sample)
#true_p_value = 1 - max_diff
#draw_graphic()
#print("true_p_value = ", true_p_value)
#statistic, p_value_ = kstest(sample, 'uniform', args=(0, 1))
#print(statistic, p_value_   )
print("ОЧЕНЬ ТОЧНОЕ расстояние = ",sqrt(n)*D_exactly)



