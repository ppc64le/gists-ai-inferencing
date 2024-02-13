import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import math
import os
import joblib
import pydot
import pickle
print(tf.__version__)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
from datetime import datetime

tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(1)

'''
with open('/tmp/udtf_features.txt', 'w') as f:
    print(tf.config.threading.get_inter_op_parallelism_threads(), file=f)
    print(tf.config.threading.get_intra_op_parallelism_threads(), file=f)
'''
seq_length = 3 # Six past transactions followed by current transaction
batch_size = 640

def fraudEncoder(X):
    return np.where(X == 'Yes', 1, 0).astype(int)

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

class full_pipeline(nzae.Ae):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def _runUdtf(self):
        
        def gen_predict_batch(tdf, mapper):
            new_df = mapper.transform(tdf).drop(['Is Fraud?'],axis=1)
            xbatch = new_df.to_numpy().reshape(batch_size, seq_length, -1)
            xbatch_t = np.transpose(xbatch, axes=(1,0,2))
            return xbatch_t

        #####################
        ### INITIALIZATON ###
        #####################
        
        model = tf.keras.models.load_model('/home/db2inst1/sqllib/function/routine/us-bank-udf/model_seq3.h5')
        mapper = joblib.load(open(os.path.join('/home/db2inst1/sqllib/function/routine/us-bank-udf/','fitted_mapper.pkl'),'rb'))

        #######################
        ### DATA COLLECTION ###
        #######################
        # Collect rows into batches
        row_list = []
        row_count = 0
        for row in self:
            # Collect everything but first element (which is select count(*))
            row_list.append(row[1:])
            row_count = row_count +1

            # Once we have all the roes, perform the data reshaping and scoring all together       
            if (row_count == row[0]) :

                # Data cleasing
                data = np.array(row_list)
                tdf = pd.DataFrame(data, columns=['Index', 'User', 'Card', 'Year', 'Month', 'Day', 'Time', 'Amount', 'Use Chip', 'Merchant Name', 'Merchant City',
                                                  'Merchant State', 'Zip', 'MCC', 'Errors?', 'Is Fraud?'])
                tdf.set_index('Index')

                tdf["Zip"] = tdf["Zip"].astype(float)
                tdf["MCC"] = tdf["MCC"].astype(int)
                tdf["Merchant Name"] = tdf["Merchant Name"].astype(str)
                tdf["Merchant City"].replace("ONLINE", " ONLINE", regex=True, inplace=True)
                tdf["Merchant State"].fillna(np.nan, inplace=True)
                tdf["Errors?"].fillna('missing_value', inplace=True)
       
                # We must use the same batch size used for model build as for scoring.  Thus this is fixed for the given model
                len_x, len_y = tdf.shape
                num_sets = len_x//seq_length
                num_batches = num_sets // batch_size 

                result = []
                 
                for i in range(num_batches):
                    batch_start = i* batch_size*seq_length
                    batch_end = (i+1) * batch_size*seq_length
                    new_tdf = tdf.iloc[batch_start:batch_end, ]
                    
                    xbatch_t = np.asarray(gen_predict_batch(new_tdf, mapper))
                    
                    prediction = model.predict(xbatch_t)
                    
                    result = [float(prediction[seq_length-1][k][0]) for k in range(batch_size)]
                
                    for j in range(len(result)):
                        # output 3 columns: index, prediction, is fraud
                        row_index = (j + 1) * seq_length - 1 + (new_tdf.shape[0] * i)
                        curr_row = new_tdf.loc[row_index]
                        self.output(int(curr_row['Index']), result[j], curr_row['Is Fraud?'])
        
        self.done()

full_pipeline.run()
