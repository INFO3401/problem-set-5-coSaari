# Utility functions for epidemiology data
import pandas as pd

# Use this to change any date information to the right type
# and to move all date data from columns to rows
def correctDateFormat(df):
    # Move date data from columns to rows. Will create two new columns (one for
    # date and one for the number of confirmed cases). Will add a new row for
    # each date x province/state
    df = df.melt(id_vars=df.columns[0:4], var_name="Date", value_name="Confirmed")
    # Convert date to a datetime object so pandas knows how to do math with it
    df["Date"] = pd.to_datetime(df["Date"])
    return df

# Helper function you can use to group the data for a given country of interestself.
# Just pass the function your dataframe and the country's name as a string
def aggregateCountry(df, country):
    data = df.loc[df["Country/Region"] == country]
    return data.groupby("Date", as_index=False).sum()



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

'''
def topCorrelation(df, tColumn, n):
    timeout = time.time() + 30
    countries = pd.unique(df["Country/Region"])

    newList = []
    empty = pd.DataFrame()
    corrDF = pd.DataFrame()
    
    for country1 in countries:
        for country2 in countries:
            if country1 != country2:
        
                x = empty.copy()
                y = empty.copy()
                z = empty.copy()
                
                x = df.loc[df["Country/Region"] == country1]
                y = df.loc[df["Country/Region"] == country2]
                
                z = pd.merge(x,y,on="Date")
                
                correlation = z[str(tColumn)+"_x"].corr(z[str(tColumn)+"_y"])
                newList.append([country1, country2, correlation])
                print(newList)
                
                #print(str(country1) +"      " +str(country2))
                #print(z[str(tColumn)+"_x"].corr(z[str(tColumn)+"_y"]))
                
                
                corrDF["Country 1"] = country1
                corrDF["Country 2"] = country2
                corrDF["Correlation"] = z[str(tColumn)+"_x"].corr(z[str(tColumn)+"_y"])
                corrDF.sort_values(by = "Correlation", ascending = False)
                
                if time.time() > timeout:
                    break
                
    print(corrDF.iloc[:n])
'''

'''n covid.py, write a function called topCorrelations that takes a 
dataframe, a target column (confirmed, deaths, recoveries, and a number n. 
The function should return the top n most correlated pairs of countries in the dataset. 
Note that your computeCorrelation function from Problem Set #4 might come in handy here. '''
'''

                        
'''