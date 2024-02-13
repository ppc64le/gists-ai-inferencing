import numpy as np
import pandas as pd
import math

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer


def fraudEncoder(X):
    return np.where(X == 'Yes', 1, 0).astype(int)


def decimalEncoder(X,length=5):
    dnew = pd.DataFrame()
    for i in range(length):
        dnew[i] = np.mod(X,10)
        X = np.floor_divide(X,10)
    return dnew


def timeEncoder(X):
    X_hm = X['Time'].str.split(':', expand=True)
    d = pd.to_datetime(dict(year=X['Year'],month=X['Month'],day=X['Day'],hour=X_hm[0],minute=X_hm[1])).astype(int)
    return pd.DataFrame(d)


def amtEncoder(X):
    amt = X.apply(lambda x: x[1:]).astype(float).map(lambda amt: max(1,amt)).map(math.log)
    return pd.DataFrame(amt)


# LabelEncoder transformer replaces categorical variables with numerical labels
# LabelBinarizer function from sklearn, which creates a column for each unique
#   value in a category, and represents membership with 1s and 0s. (1 for members,
#   0 for non members)

def get_data_frame_mapper():
    return DataFrameMapper(
        [
            ('Is Fraud?', FunctionTransformer(fraudEncoder)),
            (['Merchant State'], [
                SimpleImputer(strategy='constant'), FunctionTransformer(np.ravel),
                LabelEncoder(), FunctionTransformer(decimalEncoder), OneHotEncoder(),
            ]),
            (['Zip'], [
                SimpleImputer(strategy='constant'), FunctionTransformer(np.ravel),
                FunctionTransformer(decimalEncoder), OneHotEncoder(),
            ]),
            ('Merchant Name', [
                LabelEncoder(), FunctionTransformer(decimalEncoder), OneHotEncoder(),
            ]),
            ('Merchant City', [
                LabelEncoder(), FunctionTransformer(decimalEncoder), OneHotEncoder(),
            ]),
            ('MCC', [
                LabelEncoder(), FunctionTransformer(decimalEncoder), OneHotEncoder(),
            ]),
            (['Use Chip'], [
                SimpleImputer(strategy='constant'), LabelBinarizer(),
            ]),
            (['Errors?'], [
                SimpleImputer(strategy='constant'), LabelBinarizer(),
            ]),
            (['Year','Month','Day','Time'], [
                FunctionTransformer(timeEncoder), MinMaxScaler(),
            ]),
            ('Amount', [
                FunctionTransformer(amtEncoder), MinMaxScaler(),
            ])
        ],
        input_df=True, df_out=True,
    )
