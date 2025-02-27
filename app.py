import streamlit as st
import torch
import pennylane as qml
import numpy as np
from sklearn.preprocessing import StandardScaler
import shap
import xgboost as xgb
import pandas as pd
from datasets import load_dataset  

# Load the trained model
class QuantumFraudDetector(torch.nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnode = qml.qnn.TorchLayer(quantum_circuit, self.weight_shapes)

    def forward(self, x):
        return torch.sigmoid(self.qnode(x))

def quantum_circuit(inputs, weights):
    n_qubits = inputs.shape[1]
    qml.AngleEmbedding(features=inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

# Use st.cache_data for caching data like datasets
@st.cache_data
def load_data():
    ds = load_dataset("thomask1018/credit_card_fraud")
    data = ds['train'].to_pandas()
    return data

# Use st.cache_resource for caching models
@st.cache_resource
def load_model():
    # Here, load your trained model
    model = QuantumFraudDetector(n_qubits=4, n_layers=2)  # Example, load your actual model here
    return model

def predict_fraud(model, input_data):
    with torch.no_grad():
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output = model(input_tensor).view(-1)
        prediction = (output > 0.5).float()
        return prediction.numpy()

# Streamlit Interface
st.title("Credit Card Fraud Detection Using Quantum Machine Learning")

st.write("""
    This is a quantum fraud detection system using a quantum neural network.
    """)

# Input Data
st.sidebar.header("Input Parameters")

# Assuming 10 features selected through SHAP
input_data = []
for i in range(10):
    input_data.append(st.sidebar.number_input(f"Feature {i+1}", value=0.0, min_value=-10.0, max_value=10.0))

input_data = np.array(input_data).reshape(1, -1)

# Button to make predictions
if st.sidebar.button("Predict Fraud"):
    st.write("Processing...")
    
    # Load your trained model
    model = load_model()  # load the model here
    
    # Make Prediction
    prediction = predict_fraud(model, input_data)
    
    # Display Results
    if prediction == 1:
        st.write("This transaction is predicted to be **Fraud**.")
    else:
        st.write("This transaction is predicted to be **Not Fraud**.")
