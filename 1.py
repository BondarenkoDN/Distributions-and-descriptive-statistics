import numpy as np         # библиотека для матриц и математики
import pandas as pd        # библиотека дл работы с табличками
from scipy import stats    # модуль для работы со статистикой
from statsmodels.distributions.empirical_distribution import ECDF
# библиотеки для визуализации
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')    # стиль графиков


"""rvs сгенерирует нам выборку из распределения объёма size
   cdf вычислит для нас значение функции распределения (cumulative distribution function) в указанной точке
   pdf вычислит значение плотности распредеелния (probability density function) в указанной точке
   ppf вычислит квантиль, указанного уровня"""


norm_rv = stats.norm(loc=0, scale=1)  # задали генератор 
sample = norm_rv.rvs(1000)  # сгенерируем 1000 значений
print(sample[:10])
print(sample.shape)
print(np.mean(sample))# выборочное среднее(при больших n похоже на математическое ожидание)
print(np.var(sample))# выборочная дисперсия
print(np.std(sample)) # выборочное стандартное отклонение
print(np.median(sample))# выборочная медиана
plt.hist(sample, bins=1000)
plt.show()


norm_rv.pdf(1)
x = np.linspace(-3, 3, 100)
pdf = norm_rv.pdf(x)
plt.plot(x, pdf)
plt.ylabel('$f(x)$')# подписи осей
plt.xlabel('$x$')
# На ней же нарисуем f(1)
plt.scatter([1,2], [norm_rv.pdf(1), norm_rv.pdf(2)], color="blue")
plt.show()
print(norm_rv.cdf(1)) # узнаем значение функции распределения в точке 1


# изобразить это можно следующим образом:
plt.plot(x, pdf)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')
plt.scatter([1], [norm_rv.pdf(1)], color="blue")
xq = np.linspace(-3,1,100)
yq = norm_rv.pdf(xq)
plt.fill_between (xq,0,yq,color = 'black', alpha = 0.3)
plt.axvline(1,color = 'green', linestyle = "-.", lw = 3)
plt.show()


# построим картинку для функции распределения
cdf = norm_rv.cdf(x)
plt.plot(x, cdf)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')
plt.scatter([1], [norm_rv.cdf(1)], color="blue")
plt.show()


# вероятность попасть в кокретный отрезок от 1 до 3 
print(norm_rv.cdf(3) - norm_rv.cdf(1))
plt.plot(x, pdf)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')
plt.scatter([1, 3], [norm_rv.pdf(1), norm_rv.pdf(3)], color="blue")
xq = np.linspace(1, 3)
yq = norm_rv.pdf(xq)
plt.fill_between(xq, 0, yq, color='green', alpha=0.2)
plt.axvline(1, color='black', linestyle="--", lw=2)
plt.axvline(3, color='yellow', linestyle="--", lw=2)
plt.show()



# плотность 
plt.plot(x, pdf, lw=3)
# гистограмма, параметр density отнормировал её. 
plt.hist(sample, bins=30, density=True)
plt.ylabel('$f(x)$')
plt.xlabel('$x$')
plt.show()

# для построения ECDF используем библиотеку statsmodels
ecdf = ECDF(sample)   # строим эмпирическую функцию по выборке
plt.step(ecdf.x, ecdf.y, label='empirical CDF')
plt.plot(x, cdf, label='theoretical CDF')
plt.ylabel('$F(x)$', fontsize=20)
plt.xlabel('$x$', fontsize=20)
plt.legend(loc='upper left')
plt.show()