import json
import pandas as pd
import numpy as np
import random
from features.AddNoise import add_noise
from sklearn.model_selection import train_test_split
from models.model_utils import scale, train_model


if __name__=="__main__":
        # 1) Load config file
    with open("config/train_config.json", "r") as f:
        config = json.load(f)

    input_path   = config["inputpath"]
    test_size    = config["testdata_size"]
    lasso_alpha  = config["lasso_alpha"]
    model_path   = config["modelpath"]
    seed         = config["seed"]
    unecessary   =config["unnecessary_var"]


###Load the data
    np.random.seed(seed) 

####This code does something
    """This reads in the csv, skips the first row and also deletes the first column"""
    revenue=pd.read_csv(input_path,skiprows=1)
    if revenue.columns[0].startswith('Unnamed'):
        revenue=revenue.iloc[:,1:]
    revenue= revenue[(revenue['grossRevenue']!=0) & (revenue['expenses']!=0)]
    #print(revenue.head())
    #print(revenue.shape)

###Add noise to the data

    """This takes these features which in staging have SEVERAL duplicates.  So i just added noise to the variables"""
    noisecol=['expenses','grossRevenue','grossProfit','netProfit']
    revenue=add_noise(revenue,noisecol)


###split the data 
    """This splits the data into training and testing data"""
    X=revenue[['grossRevenue','grossProfit','netProfit']]
    y=revenue[['expenses']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

 

###Scale the data
    """This scales the data using Standard Scaler"""
    
    X_train_scaled, X_test_scaled=scale(X_train,X_test)
   # print(len(X_train_scaled))
   # print(len(X_train))



###Model the data using Lasso and the correct 
    model, metrics=train_model(X_train_scaled, y_train, X_test_scaled, y_test, alpha=lasso_alpha, model_path=model_path)
    print(f"Test MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")

  


    
   