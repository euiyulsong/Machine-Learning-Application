
# coding: utf-8

# ## 1 Active data exploration

# In[2]:


# Let's import the important packages.  Depending on how do your
# exactly proceed, you may need more.  

# Numpy is a library for working with Arrays
import numpy as np
print("Numpy version:        %6.6s" % np.__version__)

# SciPy implements many different numerical algorithms
import scipy as sp
print("SciPy version:        %6.6s" % sp.__version__)

# Pandas makes working with data tables easier
import pandas as pd
print("Pandas version:       %6.6s" % pd.__version__)

# Module for plotting
import matplotlib
print("Maplotlib version:    %6.6s" % matplotlib.__version__)
get_ipython().magic('matplotlib inline')
# needed for inline plots in notebooks
import matplotlib.pyplot as plt  

# SciKit Learn implements several Machine Learning algorithms
import sklearn
print("Scikit-Learn version: %6.6s" % sklearn.__version__)


# ## 1. Load the data

# In[3]:


from sklearn.datasets import load_boston
bdata = load_boston()


# In[4]:


print(bdata.keys())
print(bdata.feature_names)
print(bdata.data.shape)
print(bdata.target.shape)
# uncomment the following if you want to see a lengthy description of the dataset
print(bdata.DESCR)


# In[5]:


boston = pd.DataFrame(bdata.data, columns = bdata.feature_names)
boston


# ## 2. Add some (10 or so) engineered features (synthetic features) to the data. As in the previous problem set, you may use various mathematical operations on a single or multiple features to create new ones.

# In[6]:


boston['AGE*TAX'] = boston['AGE'] * boston['TAX']
boston['CRIM/AGE'] = boston['CRIM'] / boston['AGE']
boston['CRIM/B'] = boston['CRIM'] / boston['B']
boston['CRIM**2'] = boston['CRIM']**2
boston['NOX**3'] = boston['NOX']**3
boston['RM*DIS'] = boston['RM'] * boston['DIS']
boston['PTRATIO*B'] = boston['PTRATIO'] * boston['B']
boston['INDUS/NOX'] = boston['INDUS'] / boston['NOX']
boston['ZN*INDUS'] = boston['ZN'] * boston['INDUS']
boston['B/RM'] = boston['B'] / boston['RM']
boston


# ## 3. Add another set (10 or so) bogus features, variables that have no relationship whatsoever to Boston housing market. You may just pick random numbers, or numbers from irrelevant sources, such as stock prices or population of Chinese cities. Give these features distinct names (such as B1-B10) so you (and the reader) can easily recognize these later. You should have about 35 features in your data now.

# In[7]:


row = len(boston['CRIM'])
boston['B1'] = np.random.uniform(low=-1, high=3, size=(row,))
boston['B2'] = np.random.randint(10000, 100100, size=(row,))
boston['B3'] = np.random.ranf((row,)) * 4
boston['B4'] = np.round(np.random.uniform(low=4.9, high=100, size=(row,)))
boston['B5'] = np.round(np.random.uniform(low=0.5, high=5.5, size=(row,)))
boston['B6'] = np.random.randint(10, 469, size=(row,))
boston['B7'] = np.round(np.random.uniform(low=3, high=10, size=(row,)))
boston['B8'] = np.round(np.random.uniform(low=10, high=2000, size=(row,)))
boston['B9'] = np.random.uniform(low=40, high=90, size=(row,))
boston['B10'] = np.random.random_sample((row,))
boston


# ## 4. Create a summary table where you show means, ranges, and number of missings for each variable. In addition, add correlation between the price and the the variable. You may add more statistics you consider useful to this table.

# In[8]:


boston['MEDV'] = bdata.target


# In[9]:


boston


# In[10]:


numRow = len(boston['CRIM'])
exploration = boston.describe().T[['min', 'max', 'mean']]
exploration['range'] = '(' + exploration['min'].astype(str) + ', ' + exploration['max'].astype(str) + ')'
exploration['missing'] = boston.isnull().sum()
exploration.drop(['min', 'max'], axis=1)
exploration['corr'] = ""
boston
for i in boston.columns:
    exploration['corr'][i] = pd.Series.corr(boston[i], boston['MEDV'])
exploration
    


# ## 4. Graphical exploration. Make a number of scatterplots where you explore the relationship between features and the value. Include a few features you consider relevant and a few you consider irrelevant here

# In[11]:


for i in boston.columns:
    if (i != 'MEDV'):
        plt.figure(figsize = (10, 1))
        plt.scatter(boston[i], boston['MEDV'])
        plt.xlabel(i)
        plt.ylabel('MEDV')
        plt.show()



# # 2 A few simple models

# ## 2.1 Loss function

# In[12]:


import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def meanquadraticloss(lm, x, y):
    return np.square(y - lm.predict(x)).mean()


# ## 2.2 A few simple regressions

# ### 2.2.1 Create small model

# In[13]:


from sklearn.linear_model import LinearRegression

y = boston['MEDV']
x = boston[['AGE']]

lm = LinearRegression()
lm.fit(x, y)


# ### 2.2.2 10-fold cross-validate this model to get the average MSE score (your loss function).

# In[14]:


from sklearn.model_selection import cross_val_score

crossvalscore = cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss)
print('MSE score')
print(crossvalscore)
print('Average MSE score with 10-fold Cross-Validation: ', crossvalscore.mean())


# ### 2.2.3 Now take next model with 10 features. Include more features you consider relevant but also the ones you consider irrelevant. Compute 10-fold MSE for this model.

# In[15]:


x = boston[['CRIM', 'PTRATIO*B', 'INDUS', 'CHAS', 'NOX', 'RM', 'B1', 'B3', 'B9', 'TAX']]
y = boston['MEDV']

lm = LinearRegression()
lm.fit(x, y)
print('MSE score')
print(cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss))
print("Average MSE score with 10 features: ", cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss).mean())


# ### 2.2.4. Finally, include all your features and compute MSE. We call this the full model below.

# In[16]:


x = boston.drop(['MEDV'], axis=1)
y = boston['MEDV']

lm = LinearRegression()
lm.fit(x, y)

print("MSE score")
print(cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss))
print("Average MSE score with all features: ", cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss).mean())


# ### 2.2.5 Compare the results

# As MSE gets closer to zero, the model performs better result. I believe that MSE of the all the features have best model, because it has lowest average MSE score.

# # 3 Find the best model

# ## 3.1 Can we evaluate all models?

# ### 3.1.1. How many different linear regression models can you build based on the features you have (including the ones you generated)?

# In[17]:


import math
print("Number of different linear regression models: ", math.factorial(len(x.columns)))


# ### 3.1.2. Run a test: run the following loop a number of times so that the total execution time is reasonably long (at least 5 seconds) but not too long.

# In[18]:


import time

temp = boston.drop(['MEDV'], axis=1)
arange = np.arange(0, 33)

start = time.time()
for i in range(220):
    random = np.random.choice(arange, size=33, replace=False)
    numFeature = np.random.randint(low=1, high=34)
    picker = random[0:numFeature]
    
    x = temp.iloc[:, picker]
    y = boston['MEDV']
    
    lm = LinearRegression()
    lm.fit(x, y)
    
    score = cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss)
    mean = score.mean()
end = time.time()
print("Time: ", end - start)


# In[19]:


print("Time: ", ((5.5/250) * math.factorial(33)) / 31540000)


# 220 Iteration take 5.32 seconds; 33 factorial iterations take 6.07 * 10^27 years.

# ## 3.2 Foward Selection

# ### 3.2.1. Read Whitten, Frank, Hall Ch7 (on canvas, files-readings), in particular section 7.1 (attribute selection).

# ### 3.2.2. Create a series of 1-feature models and pick the best one by 10-fold CV.

# In[20]:


temp = boston.drop(['MEDV'], axis=1)
name = temp.columns

bestScore = cross_val_score(lm, temp.iloc[:, 0].values.reshape(-1, 1), boston['MEDV'], cv=10, scoring=meanquadraticloss)
bestMean = bestScore.mean()
bestName = name[0]
for i in range(33):
    x = temp.iloc[:, i].values.reshape(-1, 1)
    y = boston['MEDV']
    #print(i)
    lm = LinearRegression()
    lm.fit(x, y)
    
    score = cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss)
    mean = score.mean()
    
    if (bestMean > mean):
        bestMean = mean
        bestName = name[i]
        
print("Best/Lowest Loss: ", bestMean)
print("Best Feature Name: ", bestName)
        
    


# ### 3.2.3. Pick the feature with the lowest loss. This is your 1-feature model.

# LSTAT with 41.8289580722 loss (lowest)

# ### 3.2.4. Repeat the procedure with more features until the loss does not improve any more. This is your forward-selection model.

# In[21]:


temp = boston.drop(['MEDV'], axis=1)
name = temp.columns


arange = np.arange(0, 33)
shuffle = np.random.choice(arange, size=33, replace=False)
picker = shuffle[0:1]

x = temp.iloc[:, picker]
y = boston['MEDV']
lm = LinearRegression()
lm.fit(x, y)

score = cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss)
bestMean = score.mean()
featureNumber = 2


for i in range(1, 33):
    shuffle = np.random.choice(arange, size=33, replace=False)
    picker = shuffle[0:i]
    
    x = temp.iloc[:, picker]
    y = boston['MEDV']
    
    lm = LinearRegression()
    lm.fit(x, y)

    scoreMean = cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss).mean()
    
    if (bestMean > scoreMean):
        bestMean = scoreMean
        featureNumber = i + 1
print("Best/Lowest Loss: ", bestMean)
print("Number of features: ", featureNumber)


# In[22]:


temp = boston.drop(['MEDV'], axis=1)
name = temp.columns
x = temp.iloc[:, 0:1]
y = boston['MEDV']

lm = LinearRegression()
lm.fit(x, y)

bestMean = cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss).mean()
featureNumber = 2

for i in range(1, 33):
    x = temp.iloc[:, 0:i]
    y = boston['MEDV']
    
    lm = LinearRegression()
    lm.fit(x, y)

    scoreMean = cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss).mean()
    
    if (bestMean > scoreMean):
        bestMean = scoreMean
        featureNumber = i + 1
print("Best/Lowest Loss: ", bestMean)
print("Number of features: ", featureNumber)


# # 4 Principal components

# ## 4.1 Use raw features

# ### 4.1.1. Consult Whitten, Frank, Hall (on canvas, files-readings), in particular section 7.3.

# ### 4.1.2. Perform Principal Component Analysis on all the features in your data (except the price, medv). Extract all components (the number should equal to the number of features) and report:
# 

# In[23]:


from sklearn.decomposition import PCA

temp = boston.drop(['MEDV'], axis=1)
column = temp.columns

pca = PCA().fit(temp)
summary = pd.DataFrame(pca.explained_variance_, columns = ['Variance'], index=column)
summary['Proportional Variance'] = pca.explained_variance_ratio_
summary['Cumulative Variance'] = np.cumsum(pca.explained_variance_)
summary


# ### 4.1.3. Rotate data: rotate the original features according to the principal components. Most packages have this function built-in but you can consult Rajarman et al section 11.2.1 for details and interpretation.

# In[30]:


boston = pd.DataFrame(bdata.data, columns = bdata.feature_names)

rotation = PCA().fit_transform(boston)

for i in range(13):
    if (i != 0):
        plt.scatter(rotation[:, 0], rotation[:, i])
        plt.show()
print("Rotation Matrix: ", rotation)
print("Rotation Variance: ", np.apply_along_axis(np.var, 0, rotation))


# ### 4.1.4. Find the optimal model in rotated data: estimate the regression model explaining the housing value by the rotated features. Start with the first (most important) rotated feature and add rotated features to the model one-by-one. Each time cross-validate your result. Stop where the cross-validation score starts to deteriorate.

# In[31]:


boston = pd.DataFrame(bdata.data, columns = bdata.feature_names)
y = bdata.target
x = rotation[:, 0].reshape(-1, 1)

lm = LinearRegression()
lm.fit(x, y)
mean = cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss).mean()
resultFeature = boston.columns[0]
resultMean = mean

for i in range(13):
    x = rotation[:, i].reshape(-1, 1)
    
    lm = LinearRegression()
    lm.fit(x, y)
    mean = cross_val_score(lm, x, y, cv=10, scoring=meanquadraticloss).mean()
    
    if (resultMean > mean):
        resultMean = mean
        resultFeature = boston.columns[0]
        
print("Best Mean: ", resultMean)
print("Best Feature: ", resultFeature)


# ## 4.2 PCA on normalized data

# ### 4.2.1. Code such a function and apply this to all explanatory variables in your data. This gives you a normalized data matrix Xn.

# In[32]:


def normal(n):
    return (n - np.mean(n)) / np.std(n)


# In[33]:


for i in boston.columns:
    boston[i] = normal(boston[i])
    
boston


# ### 4.2.2 Repeat the analysis in 4.1 with normalized data.

# In[34]:


column = boston.columns

pca = PCA().fit(boston)
summary = pd.DataFrame(pca.explained_variance_, columns = ['Variance'], index=column)
summary['Proportional Variance'] = pca.explained_variance_ratio_
summary['Cumulative Variance'] = np.cumsum(pca.explained_variance_)
summary


# In[36]:


rotation = PCA().fit_transform(boston)
rotation
for i in range(13):
    if (i != 0):
        plt.scatter(rotation[:, 0], rotation[:, i])
        plt.show()
print("Rotation Matrix: ", rotation)
print("Rotation Variance: ", np.apply_along_axis(np.var, 0, rotation))



# ## 4.3 What's the best solution?

# ### Compare all your results: full model, forward selection, PCA on raw data, and PCA on normalized data. Which one is the most precise? Which one is the most compact? Which one is the easiest to do? Which one is the most straightforward to interpret?

# Most precise: based on the result, most precise result is the one with the smallest loss which is forward selection in my data, but the best is PCA normalized logically, because it normalized every rows and columns of the data, and analyze in scatter plot.
# 
# Most compact: forward selection, also because it has lowest loss, but logically it is normalized, because it normalized every rows and columns of the data, and plot it as a scatterplot.
#     
# Easiest: Full model, becasue only step I needed to do was to used cross validation and implement mean quadratic function.
# 
# Straightforward to interpret: forward selection, because you can see loss score changes for each iteration
