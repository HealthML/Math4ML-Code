import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from ridge_regression import RidgeRegression
import pdb

def load_temperature_data(year = None):
    """
    load data from a weather station in Potsdam
    
    """

    names = ['station', 'date' , 'type', 'measurement', 'e1','e2', 'E', 'e3']
    data = pd.read_csv('../datasets/weatherstations/GM000003342.csv', names = names)
    # convert the date column to datetime format
    data['date'] = pd.to_datetime(data['date'], format="%Y%m%d") # 47876 unique days
    types = data['type'].unique()

    tmax = data[data['type']=='TMAX'][['date','measurement']] # Maximum temperature (tenths of degrees C), 47876
    tmin = data[data['type']=='TMIN'][['date','measurement']] # Minimum temperature (tenths of degrees C), 47876
    prcp = data[data['type']=='PRCP'][['date','measurement']] # Precipitation (tenths of mm), 47876
    snwd = data[data['type']=='SNWD'][['date','measurement']] # Snow depth (mm), different shape
    tavg = data[data['type']=='TAVG'][['date','measurement']] # average temperature, different shape 1386
    arr = np.array([tmax.measurement.values,tmin.measurement.values, prcp.measurement.values]).T 

    df = pd.DataFrame(arr/10.0, index=tmin.date, columns=['TMAX', 'TMIN', 'PRCP']) # compile data in a dataframe and convert temperatures to degrees C, precipitation to mm

    if year is not None:
        # df = df[pd.datetime(year,1,1):pd.datetime(year,12,31)]
        start_date = datetime.datetime(year, 1, 1)
        end_date = datetime.datetime(year, 12, 31)
        df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    df['days'] = (df.index - df.index.min()).days
    return df

if __name__ == "__main__":

    year = 1900
    df = load_temperature_data(year = year)


    np.random.seed(2)
    idx = np.random.permutation(df.shape[0])

    idx_train = idx[0:100]
    idx_test = idx[100:]

    data_train = df.iloc[idx_train]
    data_test = df.iloc[idx_test]

    fit_mean = True     # fit a separate mean for y in the linear regression? 
    ridge = 1.0     # strength of the L2 penalty in ridge regression

    def plot_regression(N_train = 10):
        x_train = data_train.days.values[:N_train][:,np.newaxis] * 1.0
        y_train = data_train.TMAX.values[:N_train]

        reg = RidgeRegression(fit_mean=fit_mean, ridge=ridge)
        reg.fit(x_train, y_train)

        x_days = np.arange(366)[:,np.newaxis]
        y_days_pred = reg.pred(x_days)

        x_test = data_test.days.values[:,np.newaxis] * 1.0
        y_test = data_test.TMAX.values
        y_test_pred = reg.pred(x_test)
        print("training MSE : %.4f" % reg.mse(x_train, y_train))
        print("test MSE     : %.4f" % reg.mse(x_test, y_test))

        
        fig = plt.figure()
        plt.plot(x_train,y_train,'.')
        plt.plot(x_test,y_test,'.')
        plt.legend(["train MSE = %.2f" % reg.mse(x_train, y_train),"test MSE = %.2f" % reg.mse(x_test, y_test)])
        plt.plot(x_days,y_days_pred)
        plt.ylim([-27,39])
        plt.xlabel("day of the year")
        plt.ylabel("Maximum Temperature - degree C")
        plt.title("Year : %i        N : %i" % (year, N_train))
        
        return (fig, reg)
    plt.ion()
    N = 150
    fig, reg = plot_regression(N)
    plt.show()