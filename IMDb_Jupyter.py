import pandas as pd

df=pd.read_csv('IMDb_movies.csv')

df.head

df.isnull().sum(axis=0).sort_values(ascending=False) #Check the no. of NULL values in each column

df.isnull().sum(axis=1).sort_values(ascending=False) #Check the no. of NULL values in each row

df.isnull().sum(axis=0).sort_values(ascending=False)/len(df) #Check the fraction of values that are NULL in each column

#Removing unnecessary columns:

df.drop(['budget'],axis=1,inplace=True)

df.drop(['usa_gross_income'],axis=1,inplace=True)

df.drop(['worlwide_gross_income'],axis=1,inplace=True)

df.drop(['metascore'],axis=1,inplace=True)

df.drop(['reviews_from_critics'],axis=1,inplace=True)

df.drop(['reviews_from_users'],axis=1,inplace=True)

df.drop(['date_published'],axis=1,inplace=True)

df.shape  #There were 15 columns and 85,855 rows left

#Setting all the outliers to NaN and removing them

import numpy as np

outlier=['duration','year','votes','avg_vote']

for i in outlier:

	q1,q3=np.percentile(df[i],[20,80])

	iqr=q3­q1

	minimum=q1­(iqr*3)

	maximum=q3+(iqr*3)

	#assigning nan to the outliers
	df[i].iloc[df[df[i]<minimum].index]=np.nan

	df[i].iloc[df[df[i]>maximum].index]=np.nan

df.dropna(axis=0, inplace=True)

df.shape #There were 69,084 rows left

#Data visualization:

import seaborn as sb

sb.boxplot(x='duration',data=df)

df.hist(figsize=(15,10), color="#28abb9")

#Histogram plot of action movie ratings

acm=df[df['genre'].str.contains('Action')]

print("Histogram plot of action movie ratings: ")

sb.histplot(data=acm, x="avg_vote")

#Histogram plot of the runtimes of movies made in the USA: 

usm=df[df['country'].str.contains('USA')]

print("Histogram plot of runtimes of American movies: ")

sb.histplot(data=usm, x="duration")

#Histogram plot of the average votes for English movies:

eng=df[df['language'].str.contains('English')]

sb.histplot(data=eng,x='avg_vote')

#No. of movies in certain languages:

df.language.value_counts()

#No. of movies in certain genres:

df.genre.value_counts()

#Graph plot of the votes counted year-by-year:

sb.relplot(x='year',y='votes',kind='line',data=df)

#Average vote of movies by genre:

import matplotlib.pyplot as plt

plt.figure(figsize=(17,10))

singlegnrs=df[~df['genre'].str.contains(',')]

sb.barplot(x='genre',y='avg_vote',data=singlegnrs)

#Average vote of movies by country:

singlec=df[~df['country'].str.contains(',')]; plt.figure(figsize=(150,10))

sb.plotting_context(font_scale=72)

sb.barplot(x='country',y='avg_vote',data=singlec)

#Mean and variance of each column:

print('Duration(mean & variance):');

df.duration.mean()

# 98.4 min 

#Std. dev: 

(df.duration.std())**2

# 379.77672010710216

print('Scores: \nMean:'); 

df.avg_vote.mean()

# Mean score: 5.831211279022639

print('\nVariance')

(df.avg_vote.std())**2

# 1.4532785524683547

print('No. of votes: \nMean:')

df.votes.mean()

#Mean: 1174.2338891783916

print('\nVariance')

(df.votes.std())**2

#Variance:  3150456.1925049387

print('\nStandard deviation:')

df.votes.std()

#Standard deviation:  1774.9524479559836

print('Release year: \nMean:')

int(df.year.mean())

#Mean: 1992

print('\nVariance'); (df.year.std())**2

#Variance:  624.5936904869372

print('\nMedian:')

int(df.year.median())

#Median: 2002

#Normalization, Standardization, and hypothesis testing:

from sklearn import preprocessing

X=df[['votes','avg_vote','duration','year']]

scale=preprocessing.MinMaxScaler()

X=scale.fit_transform(X)

X=pd.DataFrame(X)

print(X.describe())

#Histogram of normalized data:

X.hist()

import random

print('Hypothesis testing: a random sample of movies will be drawn and their runtimes will be analyzed.')

print('Null hypothesis(H0): the average duration of a movie in the sample is below or equal to 100 minutes.')

print('Alternate hypothesis(H1): the average duration a movie in the sample is above 100 minutes.')

l1=random.randint(25000,30000)

l2=random.randint(45000,50000)

sample=df[l1:l2]

print('Sample: \n',sample.head(10))

print(sample.shape,' ',sample.duration.mean(),' ',sample.duration.std())

#Shape:(19865, 15). Mean: 100.38217971306318 minutes. Std. dev.: 19.751482164772277

'''The sample mean is 100.38 minutes; the sample size is 19865 movies, and the standard deviation= 19.7 5 minutes.

We will choose a significance level of 0.05. The expected z­score is 1.645.''' 

s_mean=sample.duration.mean(); s_sd=sample.duration.std(); n=19865; z=(s_mean­100)/(s_sd/(n)**0.5)

print(z)

#z=2.727170081348301

 '''The p­value is 1­0.9968=0.0032. As the null hypothesis seems very unlikely, we may reject it and conclude that the average

runtime of the movies in the set is over 100 minutes.'''


#Correlation analysis:

'''Relationship analyzed: between the average IMDB score of a movie and its duration'''

col1=df[500:600]['avg_vote']

col2=df[500:600]['duration'] 

plt.plot(col1,col2)

from scipy import stats

s,p=stats.spearmanr(col1,col2)

print('Stat: ',s,' p: ',p)

#s: 0.21157709475090614	p: 0.0345874370809851

'''p is lower than 0.05; this suggests that there is no correlation between the runtime and the average score.'''