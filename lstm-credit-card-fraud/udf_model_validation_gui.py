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

seq_length = 7 # Six past transactions followed by current transaction
batch_size = 16
total_transacations = 0

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

        model = tf.keras.models.load_model('/home/db2inst1/sqllib/function/routine/us-bank-udf/16batch_models/model_old_functions.h5')
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
                total_transacations = (row_count)/(seq_length)
                with open('/tmp/test.txt', 'w') as f:
                    print("The batch size is: ", batch_size, file=f)
                    print("The number of rows processed is: ", row_count, file=f)

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
                
                fraud_count = 0
                fraud_rows = []
                fraud_data = []  # List to store the information from tdf for fraud rows
                
                for i in range(num_batches):
                    batch_start = i* batch_size*seq_length
                    batch_end = (i+1) * batch_size*seq_length
                    new_tdf = tdf.iloc[batch_start:batch_end, ]

                    xbatch_t = np.asarray(gen_predict_batch(new_tdf, mapper))
                    
                    prediction = model.predict(xbatch_t)
                    
                    result = [float(prediction[seq_length-1][k][0]) for k in range(batch_size)]
                    
                    for j, res in enumerate(result):
                        if res >= 0.5:
                            fraud_count = fraud_count + 1
                            fraud_rows.append((i, j))  # Store the batch and index of the fraud prediction
                    
                    for j in range(len(result)):
                        row_index = (j + 1) * seq_length - 1 + (new_tdf.shape[0] * i)
                        curr_row = new_tdf.loc[row_index]
                        # Collect actual fraud rows
                        if curr_row['Is Fraud?'] == 'Yes':
                            fraud_data.append(curr_row)
                        #self.output(result[j])
                        self.output(int(curr_row['Index']), result[j], curr_row['Is Fraud?'])    

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        
        with open('/tmp/test.txt', 'a') as f:
            print("The fraud prediction count is: ", fraud_count, file=f)
            print("Fraud rows:", fraud_rows, file=f)
            
            predicted_fraud_data_indices = []
            
            for batch, index in fraud_rows:
                # Get the start and end indices of the 7-row sequence
                seq_start = (batch * batch_size + index) * seq_length
                seq_end = (batch * batch_size + index + 1) * seq_length
                # Retrieve the corresponding sequence from the new_tdf DataFrame
                fraud_sequence = tdf.iloc[seq_start:seq_end, :]
                last_index = fraud_sequence.iloc[-1]['Index']
                predicted_fraud_data_indices.append(last_index)
                print("Batch:", batch, "Index:", index, "Sequence:\n", fraud_sequence, file=f)

            print("Actual fraud data: ", file=f)
         
            actual_fraud_data_indices = []
            for row in fraud_data:
                print(row, file=f)
                actual_fraud_data_indices.append(row['Index'])

            #predicted_fraud_data_indices = [row[-1] for row in fraud_rows]
            #actual_fraud_data_indices = fraud_data['Index'].tolist()
            print("Actual fraud data: ", actual_fraud_data_indices,file=f)
            print("Pred fraud data: ", predicted_fraud_data_indices, file=f)
            false_positives = len(set(predicted_fraud_data_indices) - set(actual_fraud_data_indices))
            false_negatives = len(set(actual_fraud_data_indices) - set(predicted_fraud_data_indices))
            true_positives = len(set(actual_fraud_data_indices).intersection(set(predicted_fraud_data_indices)))
            true_negatives = total_transacations - (true_positives + false_negatives + false_positives)
            print("Total transactions: ", int(total_transacations), file=f)
            print("False positives: ", false_positives,file=f)
            print("False negatives: ", false_negatives, file=f)
            print("True positives: ", true_positives, file=f)
            print("True negatives: ", int(true_negatives), file=f)
            print("Fraud_data: ", fraud_data, file=f)
        
        with open('/tmp/predicted_fraud.txt', 'w') as f:
            predicted_fraud_data = pd.DataFrame(columns=['Index', 'User', 'Card', 'Year', 'Month', 'Day', 'Time', 'Amount', 'Use Chip', 'Merchant Name', 'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?', 'Is Fraud?'])

            for batch, index in fraud_rows:
                # Get the start and end indices of the 7-row sequence
                seq_start = (batch * batch_size + index) * seq_length
                seq_end = (batch * batch_size + index + 1) * seq_length
                # Retrieve the corresponding sequence from the new_tdf DataFrame
                fraud_sequence = tdf.iloc[seq_start:seq_end, :]
                last_entry = fraud_sequence.iloc[-1].copy()
                predicted_fraud_data = predicted_fraud_data.append(last_entry)

            print("predicted fraud data: ", file=f)
            selected_columns = ['Year', 'Month', 'Day', 'Time', 'Amount', 'Merchant City', 'Merchant State', 'Is Fraud?']
            selected_data = predicted_fraud_data[selected_columns]

            # Rename the "Is Fraud?" column to "is_Fraud?"
            selected_data = selected_data.rename(columns={'Is Fraud?': 'is_Fraud?'})

            # Combine "Merchant City" and "Merchant State" with a comma separator
            selected_data['Location'] = selected_data['Merchant City'] + ',' + selected_data['Merchant State']

            # Drop the individual "Merchant City" and "Merchant State" columns
            selected_data = selected_data.drop(['Merchant City', 'Merchant State'], axis=1)

            # Rename the "Location" column
            selected_data = selected_data.rename(columns={'Location': 'Location'})

            print(selected_data, file=f)


        """from dash import Dash, html, dcc
        import plotly.graph_objects as go
        app = Dash(__name__)

        def extract_integers(prefix):
            integers = []
            with open('/tmp/test.txt', 'r') as f:
                for line in f:
                    if line.startswith(prefix):
                        integer_str = line[len(prefix):].strip()
                        try:
                            integer = int(integer_str)
                            integers.append(integer)
                        except ValueError:
                            pass
            return integers

        prefixes = ["Total transactions: ", "True positives: ", "True negatives: ", "False negatives: ", "False positives: "]

        data = []
        for prefix in prefixes:
            integers = extract_integers(prefix)
            data.extend([(prefix, value) for value in integers])

        df = pd.DataFrame(data, columns=["Metrics", "Value"])

        fig = go.Figure(data=go.Bar(x=df["Metrics"], y=df["Value"], name="Value"))

        fig.update_layout(
            title={
                'text': "Metrics for Fraud Analytics Workload",
                'x': 0.5,  # Center the title horizontally
                'y': 0.95  # Adjust the vertical position of the title
            },
            xaxis_title="Metrics",
            yaxis_title="Value",
            font=dict(
                family="Arial, sans-serif",
                size=14,
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            bargap=0.2,
        )

        fig.update_traces(marker_color="#1f77b4", marker_line_color="#1f77b4", marker_line_width=1.5)

        # Define the column headings
        column_headings = ['Index', 'Year', 'Month', 'Day', 'Time', 'Amount', 'is_Fraud?', 'Location']

        # Read the file and skip the first row
        predicted_frauds = pd.read_csv('/tmp/predicted_fraud.txt', delim_whitespace=True, names=column_headings, skiprows=2)

        # Print the DataFrame
        print(predicted_frauds)

        import dash_table

        # Define the layout
        app.layout = html.Div(children=[
            html.H1(children='Metrics for Fraud Analytics Workload'),

            dcc.Graph(
                id='example-graph',
                figure=fig
            ),

            html.H2("Predicted Frauds"),
            dash_table.DataTable(
                id='frauds-table',
                columns=[{'name': col, 'id': col} for col in predicted_frauds.columns],
                data=predicted_frauds.to_dict('records'),
                style_table={'overflowX': 'auto'}
            )
        ])

        if __name__ == '__main__':
            app.run_server(host='0.0.0.0', port=8050, debug=True)"""
        
        self.done()


full_pipeline.run()
