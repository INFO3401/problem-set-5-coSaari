from utils import *
from covid import *
import matplotlib.pyplot as plt
import seaborn as sb
import time

#%run utils.ipynb
#%run covid.ipynb

# Load in the data from the GitHub repository and reformat it for use with our
# statistical libraries. Note: this depends on your repository being in the same
# directory as your data.
#df = loadAndCleanData("COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
df = loadAndCleanData("time_series_19-covid-Confirmed.csv")
df = correctDateFormat(df)

df1 = loadAndCleanData("time_series_19-covid-Deaths.csv")
df1 = correctDateFormat(df1)
df1["Deaths"] = df1["Confirmed"]

df2 = loadAndCleanData("time_series_19-covid-Recovered.csv")
df2 = correctDateFormat(df2)
df2["Recovered"] = df2["Confirmed"]

print(df)
print(df1)
print(df2)

def mergeData(data1, data2, column):
    data = []
    for i in data2[column]:
        data.append(i)
    data1[column] = data
    return data1

print(mergeData(df, df1, "Deaths"))
mergedDF = mergeData(x, df2, "Recovered")
print("This is the merged Dataframe")


def plotTimeline(data, time_col, val_col):
    data.plot.line(x=time_col, y=val_col)
    plt.show()


plotTimeline(mergedDF, "Date", "Confirmed")
plotTimeline(mergedDF, "Date", "Deaths")
plotTimeline(mergedDF, "Date", "Recovered")


def plotMultipleTimelines(data, time_col, val_col, cat_col):
    plt.style.use('ggplot')
    data.plot.line(x=time_col, y=[val_col, cat_col], figsize = (10,7))
    #data.plot(x=time_col, y=[val_col,cat_col], kind="line")
    plt.show()    
    
    
plotMultipleTimelines(mergedDF, "Date", "Recovered", "Deaths")
plotMultipleTimelines(mergedDF, "Date", "Confirmed", "Deaths")
plotMultipleTimelines(mergedDF, "Date", "Confirmed", "Recovered")



def topCorrelation(df, tColumn, n):
    df1 = df.pivot_table(values=tColumn, index = "Date", columns = "Country/Region", aggfunc = "first")
    df1 = df1.corr()
    countries = df2.columns
    repeat = []
    repeat2 = []

    for country1 in countries:
        for country2 in countries:
            if country1 != country2:
                if [country1,country2] not in repeat and [country2,country1] not in repeat:
                    repeat.append([country1,country2])
                    repeat2.append(df1[country1][country2])

    df2 = pd.DataFrame(list(zip(repeat,repeat2)), columns = ["Pairs", "Corr"])
    df2.sort_values(by="Corr", inplace=True, ascending=False)
    return df2.iloc[:number]


topCorrelation(mergedDF, "Confirmed", 5)
topCorrelation(mergedDF, "Recovered", 5)
topCorrelation(mergedDF, "Deaths", 5)


#Top 5 Correlations : Plots
#Germny / Spain
x = mergedDF.loc[mergedDF["Country/Region"] == "Germany"]
y = mergedDF.loc[mergedDF["Country/Region"] == "Spain"]
z = pd.merge(x,y,on="Date")

plotMultipleTimelines(mergedDF, "Date", z["Confirmed_x"], z["Confirmed_y"])
runTemporalLinearRegression(x, "Date", z["Confirmed_x"])
runTemporalLinearRegression(x, "Date", z["Confirmed_y"])


#Czechia / Germany
x = mergedDF.loc[mergedDF["Country/Region"] == "Czechia"]
y = mergedDF.loc[mergedDF["Country/Region"] == "Germany"]
z = pd.merge(y,y,on="Date")

plotMultipleTimelines(mergedDF, "Date", z["Confirmed_x"], z["Confirmed_y"])
runTemporalLinearRegression(x, "Date", z["Confirmed_x"])
runTemporalLinearRegression(x, "Date", z["Confirmed_y"])


#Czechia / Spain
x = mergedDF.loc[mergedDF["Country/Region"] == "Czechia"]
y = mergedDF.loc[mergedDF["Country/Region"] == "Spain"]
z = pd.merge(y,y,on="Date")

plotMultipleTimelines(mergedDF, "Date", z["Confirmed_x"], z["Confirmed_y"])
runTemporalLinearRegression(x, "Date", z["Confirmed_x"])
runTemporalLinearRegression(x, "Date", z["Confirmed_y"])


#Morocco / Romania
x = mergedDF.loc[mergedDF["Country/Region"] == "Morocco"]
y = mergedDF.loc[mergedDF["Country/Region"] == "Romania"]
z = pd.merge(y,y,on="Date")

plotMultipleTimelines(mergedDF, "Date", z["Confirmed_x"], z["Confirmed_y"])
runTemporalLinearRegression(x, "Date", z["Confirmed_x"])
runTemporalLinearRegression(x, "Date", z["Confirmed_y"])


#Belgium / Portugal
x = mergedDF.loc[mergedDF["Country/Region"] == "Belgium"]
y = mergedDF.loc[mergedDF["Country/Region"] == "Portugal"]
z = pd.merge(y,y,on="Date")

plotMultipleTimelines(mergedDF, "Date", z["Confirmed_x"], z["Confirmed_y"])
runTemporalLinearRegression(x, "Date", z["Confirmed_x"])
runTemporalLinearRegression(x, "Date", z["Confirmed_y"])

