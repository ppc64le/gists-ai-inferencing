import numpy as np
import pickle
import sklearn
import warnings
from flask import Flask, request, jsonify
import time as imp_time
# Suppress all warnings
warnings.filterwarnings("ignore")
print(sklearn.__version__)

# Initialize Flask app
app = Flask(__name__)

class RiskPredictor:
    def __init__(self):
        self.enc = pickle.load(open('./model/risk_encoder.p', 'rb'))
        self.model = np.load("./model/risk_model.npy")

    def predict(self, example):
        vec_st = imp_time.time()
        vec = self.vectorize(example)
        vec_end = imp_time.time()
        print("The vectorize took: ", vec_end - vec_st)
        prob = 1.0 / (1.0 + np.exp(-self.model[0] - vec.dot(self.model[1:])))
        return prob[0]

    def vectorize(self, example):
        #x = example[:14]
        x = []
        x.append(int(example[0]))   # TRANS_ID
        x.append(int(example[1]))   # PRODUCT_ID
        x.append(int(example[2]))   # LINE_NUMBER
        x.append(int(example[3]))   # QUANTITY
        x.append(float(example[4]))   # PRICE
        x.append(float(example[5]))   # TAX
        x.append(int(example[6]))   # CUSTOMER_ID
        x.append(int(example[7]))   # MFR_ID
        x.append(example[8])   # PROD_NAME
        x.append(int(example[9]))   # PROD_CATEGORY
        x.append(example[10])   # PROD_SIZE
        x.append(float(example[11]))   # RETAIL_PRICE
        x.append(float(example[12]))   # RISK_RATING
        x.append(example[13])   # INCOME_BAND
        x.append(int(example[14]))   # CREDIT_LIMIT
        return self.enc.transform([x])


risk_predictor = RiskPredictor()

@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    total_time_st = imp_time.time()
    data = request.json
    example = data['example']
    prediction = risk_predictor.predict(example)
    total_time_ed = imp_time.time()
    print("The total time took: ", total_time_ed - total_time_st)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")

"""example = [37899722504, 81533243, 2, 3, 246.643, 0.05, 15791552, 19366186, "Product #81533243", 10, "medium", 78.2992, 0.0, "10000-50000$", 2500]

risk_probability = risk_predictor.predict(example)
print("Risk Probability:", risk_probability)"""


"""
import pandas as pd

# Load the CSV file
data = pd.read_csv("/home/skaif/icp_flask/transaction_20kln.csv")

# Initialize the RiskPredictor
risk_predictor = RiskPredictor()

# Iterate over each row in the CSV file
for index, row in data.iterrows():
    # Extract the features from the row
    example = [
        row[0],  # TRANS_ID
        row[1],  # PRODUCT_ID
        row[2],  # LINE_NUMBER
        row[3],  # QUANTITY
        row[4],  # PRICE
        row[5],  # TAX
        row[6],  # CUSTOMER_ID
        row[7],  # MFR_ID
        row[8],  # PROD_NAME
        row[9],  # PROD_CATEGORY
        row[10], # PROD_SIZE
        row[11], # RETAIL_PRICE
        row[12], # RISK_RATING
        row[13], # INCOME_BAND
        row[14]  # CREDIT_LIMIT
    ]

    # Make prediction
    prediction = risk_predictor.predict(example)

    # Print the prediction
    print("Prediction for row", index, ":", prediction)
"""
