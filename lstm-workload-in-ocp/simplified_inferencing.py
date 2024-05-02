import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import math
import os
import joblib
import pydot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
print(tf.__version__)

seq_length = 7 # Six past transactions followed by current transaction
batch_size = 1

save_dir = ''
#model = tf.keras.models.load_model('/home/skaif/US-Bank-Workload-MMA/batch16/model_batch16.h5')
model = tf.keras.models.load_model('./extra/model_1batch.h5')

def timeEncoder(X):
    X_hm = X['Time'].str.split(':', expand=True)
    d = pd.to_datetime(dict(year=X['Year'],month=X['Month'],day=X['Day'],hour=X_hm[0],minute=X_hm[1])).astype(int)
    return pd.DataFrame(d)

def amtEncoder(X):
    amt = X.apply(lambda x: x[1:]).astype(float).map(lambda amt: max(1,amt)).map(math.log)
    return pd.DataFrame(amt)

def decimalEncoder(X,length=5):
    dnew = pd.DataFrame()
    for i in range(length):
        dnew[i] = np.mod(X,10)
        X = np.floor_divide(X,10)
    return dnew

def fraudEncoder(X):
    return np.where(X == 'Yes', 1, 0).astype(int)

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer

mapper = joblib.load(open('./extra/fitted_mapper.pkl', 'rb'))

tdf = pd.read_csv('./extra/new_indices.csv')

tdf['Merchant Name'] = tdf['Merchant Name'].astype(str)
tdf.sort_values(by=['User','Card'], inplace=True)
tdf.reset_index(inplace=True, drop=True)
print (tdf.info())

# Get first of each User-Card combination
first = tdf[['User','Card']].drop_duplicates()
f = np.array(first.index)

# Drop the first N transactions
drop_list = np.concatenate([np.arange(x,x + seq_length - 1) for x in f])
index_list = np.setdiff1d(tdf.index.values,drop_list)

print("\nFull test")

def gen_predict_batch(tdf, mapper):
    new_df = mapper.transform(tdf).drop(['Is Fraud?'],axis=1)
    xbatch = new_df.to_numpy().reshape(batch_size, seq_length, -1)
    xbatch_t = np.transpose(xbatch, axes=(1,0,2))
    return xbatch_t

len_x, len_y = tdf.shape
num_sets = len_x//seq_length
num_batches = num_sets // batch_size

def predict_user_card_combination(tdf, mapper, model, user, card):

    batch_predictions = []
    # Loop to collect the rows for the specified user-card combination in batches of 7
    for i in range(0, len(tdf), seq_length):
        batch_data = tdf.iloc[i:i+seq_length]  # Get a batch of 7 user-card combinations
        xbatch_t = np.asarray(gen_predict_batch(batch_data, mapper))

        # Make predictions for this batch
        predictions = model.predict(xbatch_t)

        result = float(predictions[seq_length - 1][0][0])
        batch_predictions.append(result)

    print("Number of predictions for 7 transaction sequences: ", len(batch_predictions))
    return batch_predictions

def add_row_to_dataframe(dataframe, user, card, year, month, day, time, amount, use_chip, merchant_name, merchant_city, merchant_state, zip_code, mcc):
    new_row = {
        'User': user,
        'Card': card,
        'Year': year,
        'Month': month,
        'Day': day,
        'Time': time,
        'Amount': amount,
        'Use Chip': use_chip,
        'Merchant Name': merchant_name,
        'Merchant City': merchant_city,
        'Merchant State': merchant_state,
        'Zip': zip_code,
        'MCC': mcc
    }

    dataframe = pd.concat([dataframe, pd.DataFrame([new_row])], ignore_index=True)

    return dataframe


user_id = 0
card = 0
year, month, day = 2002, 9, 13
time = '06:37'
amount = "$44.41"
use_chip = "Swipe Transaction"
merchant_name = "-34551508091458520"
merchant_city = "La Verne"
merchant_state = "CA"
zip_code = 91750
mcc = 5912

"""
user_id = 29
card = 3
year, month, day = 2018, 12, 19
time = '12:38'
amount = "$1.88"
use_chip = "Chip Transaction"
merchant_name = "6051395022895754231"
merchant_city = "Rome"
merchant_state = "Italy"
zip_code = None
mcc = 5310
num_rows_to_fetch = 6
"""

# Filter data based on user_id, card, year, and month
tdf = tdf[(tdf['User'] == user_id) &
                    (tdf['Card'] == card) &
                    (tdf['Year'] == year) &
                    (tdf['Month'] <= month)]

# Limit the number of rows to 6
tdf = tdf.head(6)

tdf = add_row_to_dataframe(tdf, user_id, card, year, month, day, time,
                               amount, use_chip, merchant_name, merchant_city,
                               merchant_state, zip_code, mcc)

print(tdf)
prediction_result = predict_user_card_combination(tdf, mapper, model, user_id, card)
print(prediction_result)

"""
for i in range(num_batches):
    batch_start = i* batch_size*seq_length
    batch_end = (i+1) * batch_size*seq_length
    new_tdf = tdf.iloc[batch_start:batch_end, ]
    xbatch_t = np.asarray(gen_predict_batch(new_tdf, mapper))
    prediction = model.predict(xbatch_t)
    result = [float(prediction[seq_length-1][k][0]) for k in range(batch_size)]
    for j in range(len(result)):
        print(result[j])
"""
