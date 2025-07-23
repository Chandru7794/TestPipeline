import random
import pandas as pd
import numpy as np

def add_noise(df, columns, seed=100) -> pd.DataFrame:
    #This code adds some noise to some selected variables
    np.random.seed(seed) #this seeds Numpy's RNG
    for col in columns:
        noise=np.random.uniform(-0.05,0.05,size=len(df))
        df[col] = df[col] * (1 + noise)
    return df


