import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import math
import os
import joblib
import pydot
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Test to see if libraries loaded")

from dash import Dash, html, dcc
import plotly.graph_objects as go
import pandas as pd
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
    app.run_server(host='0.0.0.0', port=8050, debug=True)

