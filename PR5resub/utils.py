import pandas as pd

import math
import numpy as np
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Graphics Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# Import data handling libraries
#import datetime as dt
#from pandas.core import datetools

def helloWorld():
    print("Hello, World!")

def loadAndCleanData(filename):
    data = pd.read_csv(filename)
    data.fillna(0)
    #print(data)
    return data

def computeProbability(feature, bin, data):
    # Count the number of datapoints in the bin
    count = 0.0

    for i,datapoint in data.iterrows():
        # See if the data is in the right bin
        if datapoint[feature] >= bin[0] and datapoint[feature] < bin[1]:
            count += 1

    # Count the total number of datapoints
    totalData = len(data)

    # Divide the number of people in the bin by the total number of people
    probability = count / totalData

    # Return the result
    return probability

def computeConfidenceInterval(data):
      # Confidence intervals
        npArray = 1.0 * np.array(data)
        stdErr = scipy.stats.sem(npArray)
        n = len(data)
        return stdErr * scipy.stats.t.ppf((1+.95)/2.0, n - 1)

def getEffectSize(d1,d2):
    m1 = d1.mean()
    m2 = d2.mean()
    s1 = d1.std()
    s2 = d2.std()

    return (m1 - m2) / math.sqrt((math.pow(s1, 3) + math.pow(s2, 3)) / 2.0)

def runTTest(d1,d2):
    return scipy.stats.ttest_ind(d1,d2)

# pip install statsmodels
# vars is a string with our independent and dependent variables
# " dvs ~ ivs"
def runANOVA(dataframe, vars):
    model = ols(vars, data=dataframe).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    return aov_table

# Plot a timeline of my data
def plotTimeline(data, time_col, val_col):
    data.plot.line(x=time_col, y=val_col)
    plt.show()

    
# Plot a timeline of my data broken down by each category (cat_col)
def plotMultipleTimelines(data, time_col, val_col, cat_col):
    plt.style.use('ggplot')
    data.plot.line(x=time_col, y=[val_col, cat_col], figsize = (10,7))
    #data.plot(x=time_col, y=[val_col,cat_col], kind="line")
    plt.show()      
    
    
# Run a linear regression over the data. Models an equation
# as y = mx + b and returns the list [m, b].
def runTemporalLinearRegression(data, x, y):
    # Format our data for sklean by reshaping from columns to np arrays
    x_col = data[x].map(dt.datetime.toordinal).values.reshape(-1,1)
    y_col = data[y].values.reshape(-1, 1)

    # Run the regression using an sklearn regression object
    regr = LinearRegression()
    regr.fit(x_col, y_col)

    # Compute the R2 score and print it. Good scores are close to 1
    y_hat = regr.predict(x_col)
    fitScore = r2_score(y_col, y_hat)
    print("Linear Regression Fit: " + str(fitScore))

    # Plot linear regression against data. This will let us visually judge whether
    # or not our model is any good. With small data, a high R2 doesn't always mean
    # a good model: we can use our intuition as well.
    #plt.scatter(data[x], y_col, color='lightblue')
    #plt.plot(data[x], y_hat, color='red', linewidth=2)
    #plt.show()

    # y = mx + b
    # Return m and b
    return [regr.coef_[0][0], regr.intercept_[0]]

def logistic(x, x0, m, b):
    y = 1.0 / (1.0 + np.exp(-m*(x - x0) + b))
    return (y)

def runTemporalLogisticRegression(data, x, y):
    x_col = data[x].map(dt.datetime.toordinal)
    y_col = data[y]

    #giving curve fit to start with
    p0 = [np.media(x_col), 1, min(y_col)]
    paras, pcov = curve_fit(logistic, x_col, y_col, p0)

    #show fit
    plt.scatter(data[x], y_col, color = 'lightblue')
    plt.plot(data[x], logistic(x_col, params[0], params[1], params[2], color='red', linewidth=2))
    plt.show()
    
    
    
def mergeData(data1, data2, column):
    data = []
    for i in data2[column]:
        data.append(i)
    data1[column] = data
    return data1
