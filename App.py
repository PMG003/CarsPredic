import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pickle

# Load the model
with open('random_forest_classifier.pkl', 'rb') as f:
    model = pickle.load(f)


# Load and prepare data from the image
car_data = {
    'Brand': ['Kia', 'Chevrolet', 'Mercedes', 'Audi', 'Volkswagen'],
    'Year': [2020, 2012, 2020, 2023, 2003],
    'Engine_Size': [4.2, 2.0, 4.2, 2.0, 2.6],
    'Fuel_Type': ['Diesel', 'Hybrid', 'Diesel', 'Electric', 'Hybrid'],
    'Transmission': ['Manual', 'Automatic', 'Automatic', 'Manual', 'Semi-Automatic'],
    'Mileage': [289944, 5356, 231440, 160971, 286618],
    'Doors': [3, 2, 4, 2, 3],
    'Owner_Count': [5, 3, 2, 1, 3],
    'Price': [8501, 12092, 11171, 11780, 2867]
}
df = pd.DataFrame(car_data)

# Feature engineering
features = ['Brand', 'Year', 'Engine_Size', 'Fuel_Type', 'Transmission', 'Mileage', 'Doors', 'Owner_Count']
X = df[features]
y = df['Price']

# Create preprocessing and modeling pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Brand', 'Fuel_Type', 'Transmission'])
    ],
    remainder='passthrough'
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
model.fit(X, y)

# Streamlit interface
st.title("🚗 Car Price Prediction App")
st.write("Predict a vehicle's market value using machine learning")

# Input widgets
col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", df['Brand'].unique())
    fuel_type = st.selectbox("Fuel Type", df['Fuel_Type'].unique())
    transmission = st.selectbox("Transmission", df['Transmission'].unique())
    doors = st.selectbox("Number of Doors", [2, 3, 4])

with col2:
    year = st.slider("Manufacturing Year", 2000, 2023, 2020)
    engine_size = st.slider("Engine Size (L)", 1.0, 5.0, 2.0)
    mileage = st.number_input("Mileage (km)", min_value=0, value=50000)
    owners = st.number_input("Previous Owners", min_value=0, max_value=10, value=1)

# Prediction button
if st.button("Estimate Price"):
    input_df = pd.DataFrame([[
        brand,
        year,
        engine_size,
        fuel_type,
        transmission,
        mileage,
        doors,
        owners
    ]], columns=features)
    
    prediction = model.predict(input_df)[0]
    st.subheader(f"Estimated Value: ${prediction:,.0f}")
    st.write("*Note: This prediction is based on limited data and should be used for demonstration purposes only.*")
