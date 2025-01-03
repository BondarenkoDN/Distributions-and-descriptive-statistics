import numpy as np         # библиотека для матриц и математики
import pandas as pd        # библиотека для работы с табличками
from scipy import stats    # модуль для работы со статистикой
# библиотеки для визуализации
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')    # стиль графиков

df = pd.read_csv("youtube_data_short.csv", sep = "\t")
print(df.head())
print(df.dtypes)

"""Описание колонок:
title - название видео
commentCount - число комментариев
dislikeCount - число дизлайков
likeCount - число лайков
music_style - музыкальный стиль
performer - исполнитель
viewCount - число просмотров
commentators_uniq - число уникальных комментаторов
comments_obscene_cnt - число комментариев
video_age - возраст видео в днях
"""
# общая описательная статистика
print( df.describe())
# одно и тоже через pandas и numpy
print(np.mean(df.likeCount))
print(df.likeCount.mean())
# Описательные статистики
print(df.likeCount.median())
print(df.likeCount.max())
print(df['likeCount'].min())
print(df[df.likeCount == df.likeCount.max()])#срез по выбранному фильтру
#Гисторграмма
print(df['likeCount'].hist(bins=50, density=True))
plt.show() 
#Прологарифмированная гисторграмма
print(df.likeCount.hist(bins = 50, log = True))
plt.show() 
print(df.video_age.hist(bins = 100, density = True))
print(df.video_age.plot(kind = 'kde', linewidth = 4))
plt.show()
columns = ['viewCount', 'likeCount', 'dislikeCount'] 
df_log = df[columns].apply(lambda x: np.log(x + 1)) # снова прологарифмируем
df_log['music_style'] = df['music_style']
sns.boxplot(x='music_style', y='likeCount', data= df_log)
plt.xlabel('музыкальный стиль')
plt.ylabel('логарифм числа лайков')
plt.show() 

print(df.likeCount.quantile(0.99))
print(df[df.likeCount > df.likeCount.quantile(0.99)])#срез таблицы с фильтром с сохранением всех колонок
print(df[df.likeCount > df.likeCount.quantile(0.99)].music_style)#срез таблицы с фильтром с выводом одной колонки
print(df[df.likeCount > df.likeCount.quantile(0.99)].music_style).value_counts()# группировка срезанной таблицы по жанрам

print(df.groupby(['music_style'])[['likeCount']].agg(['mean']))
print(df.title.apply(len)[:5])# вывод первых 5 колонок длины названий песен
print(df.title.apply(len).mean())
