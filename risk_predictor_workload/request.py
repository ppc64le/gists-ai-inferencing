import requests
import json

example = [37899722504, 81533243, 2, 3, 246.643, 0.05, 15791552, 19366186, "Product #81533243", 10, "medium", 78.2992, 0.0, "10000-50000$", 2500]

# Define the data payload
data = {'example': example}

# Send a POST request to the Flask app
response = requests.post('http://localhost:5000/predict_risk', json=data)

# Print the response
print("Prediction:", response.json()['prediction'])

