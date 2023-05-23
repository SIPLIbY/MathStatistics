import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
import pandas as pd
from math import sqrt
import scipy.stats as st
from scipy.stats import kstwobign
from scipy.stats import norm, t, chi2, uniform, f

X_2 = np.asarray([0.120, -0.479, 2.179, 1.068, 1.109, 1.094, 1.349, -0.250, 1.216, 2.091,
                1.169, 1.838, 1.054, 2.512, 1.026, 1.512, 1.236, 0.828, 1.532, 1.531,
                1.149, 1.018, 1.463, 0.667, 0.792, 0.632, 0.207, 1.678, 0.779, 0.590])

X_1 = np.asarray([0.936, 0.483, 1.371, 0.253, 0.515, 0.339, 0.356, 1.505, 0.743, 0.538,
                 1.522, 0.345, 2.233, 1.081, 0.581, 1.672, 2.230, 0.436, 0.902, 2.117])
#X_1 = np.random.normal(1, 0.5, 100)
#X_2 = np.random.normal(1, 0.6, 100)
n_1 = X_1.size
n_2 = X_2.size
X_mean_1 = np.average(X_1)
X_mean_2 = np.average(X_2)
#print(X_mean_1, X_mean_2)
#определяем эпсилон и находим выборочные смещенные и несмещенные дисперсии
eps = 0.13
S_1 = np.average(X_1 * X_1) - X_mean_1 ** 2
S_2 = np.average(X_2 * X_2) - X_mean_2 ** 2
So_1 = (n_1/(n_1-1))*S_1
So_2 = n_2/(n_2-1)*S_2
#Сначала используем F-test, т.к необходимо убедиться, что дисперсии выборок равны
F = (So_1 / So_2)

#Находим квантиль
quant_f_right = f.ppf(1-eps/2, n_1-1, n_2-1)
quant_f_left = f.ppf(eps/2, n_1-1, n_2-1)
#Проверка что критическое значение меньше чем, статистика
if quant_f_left < F and F < quant_f_right:
    print("F-test passed, дисперсии выборок равны между собой"); print(f'F = {F}, Критическое значение при уровне значимости {1-eps} = ({quant_f_left},{quant_f_right})')
else: print('F-test не пройден.');print(f'Значение статистики = {F}, Критическое значение при уровне значимости {1-eps} = ({quant_f_left},{quant_f_right})')
#Просто по приколу найдем РДУЗ
#p_value_F = 1-f.cdf(F, n_1-1, n_2-1)
#print("РДУЗ = ", p_value_F)
print("Далее T-test")

#T-test

T = sqrt(n_1*n_2/(n_1+n_2))*abs(X_mean_1 - X_mean_2) / (sqrt(((n_1-1)*So_1 + (n_2-1)*So_2) / (n_1+n_2-2)))
quant_t = t.ppf(1-eps/2, n_1+n_2-2)
if T < quant_t:
    print("T-test passed, матожидания выборок равны между собой"); print(f'T = {T}, Критическое значение при уровне значимости {1-eps} = {quant_t}')
#p_value_T = t.sf(T, n_1+n_2-2)
#print("РДУЗ = ", p_value_T)