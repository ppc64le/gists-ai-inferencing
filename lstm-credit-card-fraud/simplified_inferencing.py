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
batch_size = 16

save_dir = ''
model = tf.keras.models.load_model('/home/vtupili/model_batch16.h5')

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

mapper = joblib.load(open(os.path.join(save_dir,'fitted_mapper_batch16.pkl'),'rb'))

tdf = pd.read_csv('new_indices.csv')
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

def gen_test_batch(df, mapper, indices, batch_size):
    rows = indices.shape[0]
    index_array = np.zeros((rows, seq_length), dtype=np.int)
    for i in range(seq_length):
        index_array[:,i] = indices + 1 - seq_length + i
    count = 0
    while (count + batch_size <= rows):
        full_df = mapper.transform(df.loc[index_array[count:count+batch_size].flatten()])
        data = full_df.drop(['Is Fraud?'],axis=1).to_numpy().reshape(batch_size, seq_length, -1)
        #targets = full_df['Is Fraud?'].to_numpy().reshape(batch_size, seq_length, 1)
        count += batch_size
        data_t = np.transpose(data, axes=(1,0,2))
        #targets_t = np.transpose(targets, axes=(1,0,2))
        yield data_t


print("\nFull test")
#test_generate = gen_test_batch(tdf,mapper,index_list,batch_size)
#model.predict(test_generate)

def gen_predict_batch(tdf, mapper):
    new_df = mapper.transform(tdf).drop(['Is Fraud?'],axis=1)
    xbatch = new_df.to_numpy().reshape(16, seq_length, -1)
    xbatch_t = np.transpose(xbatch, axes=(1,0,2))
    return xbatch_t

len_x, len_y = tdf.shape
batch_size = 16
num_sets = len_x//seq_length
num_batches = num_sets // batch_size

for i in range(num_batches):
    batch_start = i* batch_size*seq_length
    batch_end = (i+1) * batch_size*seq_length
    new_tdf = tdf.iloc[batch_start:batch_end, ]
    xbatch_t = np.asarray(gen_predict_batch(new_tdf, mapper))
    prediction = model.predict(xbatch_t)
    result = [float(prediction[seq_length-1][k][0]) for k in range(batch_size)]
    for j in range(len(result)):
        print(result[j])

