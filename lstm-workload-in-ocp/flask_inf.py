import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import math
import os
import joblib
import pydot
import warnings
from flask import Flask, request, jsonify
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
#print(tf.__version__)

app = Flask(__name__)

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
#tdf['Merchant City'].replace('ONLINE', ' ONLINE', regex=True, inplace=True)
#tdf['Merchant State'].fillna(np.nan, inplace=True)
#tdf['Zip'].fillna(np.nan, inplace=True)
tdf.sort_values(by=['User','Card'], inplace=True)
tdf.reset_index(inplace=True, drop=True)

# Get first of each User-Card combination
first = tdf[['User','Card']].drop_duplicates()
f = np.array(first.index)

# Drop the first N transactions
drop_list = np.concatenate([np.arange(x,x + seq_length - 1) for x in f])
index_list = np.setdiff1d(tdf.index.values,drop_list)

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

    """# Function to handle None or empty values
    def handle_none_or_empty(value, default):
        if value is None or not str(value).strip():
            return default
        else:
            return value

    # Replace None or empty values with default values
    new_row['Merchant State'] = handle_none_or_empty(merchant_state, 'CA')
    new_row['Zip'] = handle_none_or_empty(zip_code, -1)"""

    dataframe = pd.concat([dataframe, pd.DataFrame([new_row])], ignore_index=True)

    return dataframe

# Define the Flask endpoint
@app.route('/predict', methods=['POST'])
def predict_fraud():
    data = request.get_json()
    user_id = int(data.get('user_id'))
    card = int(data.get('card'))
    year = int(data.get('year'))
    month = int(data.get('month'))
    day = int(data.get('day'))
    time = data.get('time')
    amount = data.get('amount')
    use_chip = data.get('use_chip')
    merchant_name = data.get('merchant_name')
    merchant_city = data.get('merchant_city')
    merchant_state = data.get('merchant_state')
    zip_code = float(data.get('zip_code'))
    mcc = int(data.get('mcc'))

    tdf_filtered = tdf[(tdf['User'] == user_id) & (tdf['Card'] == card) & (tdf['Year'] == year) & (tdf['Month'] <= month)]
    tdf_sample = tdf_filtered.head(6)

    tdf_extended = add_row_to_dataframe(tdf_sample, user_id, card, year, month, day, time, amount, use_chip, merchant_name, merchant_city, merchant_state, zip_code, mcc)

    prediction_result = predict_user_card_combination(tdf_extended, mapper, model, user_id, card)
    return jsonify({'predictions': prediction_result})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")
