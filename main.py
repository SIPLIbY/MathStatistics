import numpy as np
import matplotlib.pyplot as plt
import warnings
from math import sqrt
from scipy.stats import norm as norm_distr, t as stud_distr, chi2 as hi2_distr
np.random.seed(42)
X = np.asarray([0.936, 0.483, 1.371, 0.253, 0.515, 0.339, 0.356, 1.505, 0.743, 0.538,
                1.522, 0.345, 2.233, 1.081, 0.581, 1.672, 2.230, 0.436, 0.902, 2.117,
                0.120, -0.479, 2.179, 1.068, 1.109, 1.094, 1.349, -0.250, 1.216, 2.091,
                1.169, 1.838, 1.054, 2.512, 1.026, 1.512, 1.236, 0.828, 1.532, 1.531,
                1.149, 1.018, 1.463, 0.667, 0.792, 0.632, 0.207, 1.678, 0.779, 0.590])

#X = np.random.normal(1, sqrt(0.5), size=500)


n = X.shape[0]
true_mean = 1
true_var2 = 0.5
eps = 0.13
X_mean = np.average(X)
print("X_mean = ", X_mean)
#выборочная дисперсия
sample_var = np.average(X * X) - X_mean ** 2
#несмещенная выборочная дисперсия
sample_var_0 = (n/(n-1))*sample_var
print("sample_var_0 = ", sample_var_0)
#выборочная дисперсия с известным средним
sample_var_1 = sum((X - true_mean * np.ones(n)) ** 2) / n
#print(sample_var_0, sample_var_1)
# exit()
bins = 5
#БУКВА (А)



def case_1():
    # Ищем квантили
    quant_norm = norm_distr.ppf(1 - eps / 2)
    #print("Квантиль = ", quant_norm)
    # print(tau_eps)
    left_side_a = X_mean - quant_norm * sqrt(true_var2) / sqrt(n)
    right_side_a = (quant_norm * sqrt(true_var2)) / sqrt(n) + X_mean
    interval_length_a = abs(right_side_a - left_side_a)
    print("Case 1")
    print("Interval = (", left_side_a, ',', right_side_a, ')')
    print("Interval length = ", interval_length_a)



def case_2():
    # БУКВА (Б)

    quant_stud = stud_distr.ppf(1 - eps / 2, n - 1)

    #print("Квантиль = ", quant_stud)
    # Строим интервал
    left_side_b = X_mean - (quant_stud * sqrt(sample_var_0)) / sqrt(n)
    right_side_b = X_mean + (quant_stud * sqrt(sample_var_0)) / sqrt(n)
    interval_length_b = abs(right_side_b - left_side_b)
    print('\n', "Case 2")
    print("Interval = (", left_side_b, ',', right_side_b, ')')
    print("Interval length = ", interval_length_b)

def case_3():
    hi_2 = hi2_distr(df=n)

    quant_xi_left = hi_2.ppf(1-eps/2)
    quant_xi_right = hi_2.ppf(eps/2)
    # print("Квантиль уровня 1-eps/2 = ", quant_xi_left)
    # print("Квантиль уровня eps/2 = ", quant_xi_right)

    left = n*sample_var_1 / quant_xi_left
    right = n*sample_var_1 / quant_xi_right
    interval_length = abs(left-right)
    print('\n', "Case 3")
    print("Interval = (", left, ',', right, ')')
    print("Interval length = ", interval_length)

def case_4():
    hi_2 = hi2_distr(df=n - 1)
    quant_xi_left = hi_2.ppf(1 - eps / 2)
    quant_xi_right = hi_2.ppf(eps / 2)
    # print("Квантиль уровня 1-eps/2 = ", quant_xi_left)
    # print("Квантиль уровня eps/2 = ", quant_xi_right)
    print()
    left = n * sample_var / quant_xi_left
    right = n * sample_var / quant_xi_right
    interval_length = abs(left - right)
    print('\n', "Case 4")
    print("Interval = (", left, ',', right, ')')
    print("Interval length = ", interval_length)

case_1()
case_2()
case_3()
case_4()



plt.hist(X, bins=bins, color='steelblue', edgecolor='black')

# Настройки легенды и заголовка
plt.legend(['Значения'])
plt.title('Гистограмма данных')

# Подписи осей
plt.xlabel('Значение')
plt.ylabel('Частота')

# Отображаем гистограмму
plt.show()