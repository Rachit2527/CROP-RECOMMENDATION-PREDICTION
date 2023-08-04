import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('crop_prediction_model.joblib')

def predict_crop(data):
    # Assuming 'data' is a DataFrame containing the input features
    predictions = model.predict(data)
    return predictions

def main():
    st.title('Crop Recommendation App')

    # Collect user input for the crop features
    N = st.slider('Nitrogen (N)', 0, 150, 50)
    P = st.slider('Phosphorus (P)', 0, 150, 50)
    K = st.slider('Potassium (K)', 0, 150, 50)
    temperature = st.slider('Temperature', 0.0, 40.0, 25.0)
    humidity = st.slider('Humidity', 0.0, 100.0, 50.0)
    ph = st.slider('pH', 0.0, 14.0, 7.0)
    rainfall = st.slider('Rainfall', 0.0, 400.0, 200.0)

    # Create a DataFrame with user inputs
    user_input = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    if st.button('Predict Crop'):
        # Make predictions using the trained model
        prediction = predict_crop(user_input)
        st.write(f"Predicted Crop: {prediction[0]}")

if __name__ == '__main__':
    main()
