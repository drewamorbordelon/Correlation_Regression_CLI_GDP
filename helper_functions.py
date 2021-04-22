# import numpy as np
# import statistics as stats
import matplotlib.pyplot as plt

"""
This function averages 2 periods in a time series. 
Alternatively, one could use --> df.rolling(2).mean()
"""
def smoothingMA2(df):
    data = df.copy()
    df1 = (data + data.shift(1)) / 2
    return df1



"""
Calculates the First Difference of the 
2nd Derivative (Rate of Change/Acceleration).
""" 
def RoC(df, n):   
    df1 = df.copy()
    df1 = (np.log(df1).diff(n))
    df1 = df1 - (df1.shift(1))
    df1 = df1 * 100
    return df1


"""
Only use this roc function after getting the Year over Year 
percentage change because this function does not calculate the  
log difference or np.log().diff().
"""
def roc(df, n):
    data = df.copy()
    df1 = (data - data.shift(n))
    return df1


"""
Plots line graphs 
"""
def plotLines(df, i:int, title, w_name, x_name, y_name, z_name):
# def plotLines(df, i:int, title, x_name, y_name):
# def plotLines(df, i:int, title, w_name, y_name, z_name):
    # Controls amount of data to plot
    df =  df[500:]
    
    w = df[w_name]
    x = df[x_name]
    y = df[y_name]
    z = df[z_name]
    
    # Plot style
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots(figsize=(30,15)) # chart 2:1
    
    # Sets Title of Plot
    ax.set_title(title, fontsize=32, fontweight='bold', x=.242, y=1.07)
    
    # Line plot these columns: column1, column2
    plt.plot(w.index, w, color='black', linewidth=3, label="Price-Daily")
    plt.plot(x.index, x, color='orange', linewidth=3, label="Rolling 7D Mean")
    plt.plot(y.index, y, color='red', linewidth=3, label="UpperBound Dynamic R/S")
    plt.plot(z.index, z, color='green', linewidth=3, label="LowerBound Dynamic R/S")
    # Adds a legened
    plt.legend(loc="upper left")
    
    # Turn off the spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.show()
    
    
"""
Creates the features involved in making the plots and calculations.  
This allows one to simply apply the function to multiple dataframes 
when the same analysis is needed.
"""
def createFeatures(df, prefix:str):
    df['_%mom'] = df.iloc[:,0].pct_change() * 100
    df['_%yoy'] = df.iloc[:,0].pct_change(12) * 100
    
    df['_smoothingMA'] = (df['_%yoy'] + df['_%yoy'].shift(1) + df['_%yoy'].shift(2) + df['_%yoy'].shift(3)) / 4
    df['_rocMA'] = roc(df['_smoothingMA'], 1)
    df['_smoothedRoC'] = smoothingMA2(df['_rocMA'])
    
    df = df.add_prefix(prefix)
    df.dropna(inplace=True)

    return df


"""
Applying Standard Scaler and MinMaxScaler-->(Experimental)
Standard Scaler was used going forward in this notebook (Not MinMaxScaler).
"""

def standardScaler(df, scaled_df_name:str):
    st_scaler = StandardScaler()
    scaled_df= st_scaler.fit_transform(df)
    scaled_df= pd.DataFrame(scaled_df, columns=df.columns)
    scaled_df.dropna(inplace=True)
    scaled_df.reset_index(inplace=True)
#     scaled_df = scaled_df.add_prefix(prefix)
    scaled_df_name = scaled_df.drop(['index'], axis=1)
    return scaled_df_name