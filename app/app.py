# import streamlit as st
# import joblib
# import pandas as pd
# from utils import load_model, prepare_input_data, get_recommendations

# # Load the trained model
# model = load_model('/Users/adeliajanuarto/Documents/DirectoryPython/PORTFOLIO-PROJECT/05_RECOMMENDATION-SYSTEM/RECOMMENDATION-CROP/model/clf_final.pkl')

# # Streamlit app title
# st.title('Crop Recommendation App')

# # User input for features
# st.header('Input Features')
# nitrogen = st.number_input('Nitrogen (N) in kg/ha', min_value=0.0)
# phosphorus = st.number_input('Phosphorus (P) in kg/ha', min_value=0.0)
# potassium = st.number_input('Potassium (K) in kg/ha', min_value=0.0)
# temperature = st.number_input('Temperature in Celsius', min_value=0.0)
# humidity = st.number_input('Humidity in percentage', min_value=0.0)
# ph = st.number_input('pH value of the soil', min_value=0.0)
# rainfall = st.number_input('Rainfall in mm', min_value=0.0)

# # Prepare input data
# input_data = prepare_input_data(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)

# # Button to predict
# if st.button('Recommend Crop'):
#     recommendations = get_recommendations(model, input_data)
#     st.write('Recommended Crops:')
#     st.write(recommendations)

import streamlit as st
import pandas as pd
import joblib

# Function to load model
def load_model(path):
    return joblib.load(path)

def prepare_input_data(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    data = pd.DataFrame([{
        'N': nitrogen,
        'P': phosphorus,
        'K': potassium,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }])
    return data

# Create a dictionary for label encoding
label_decoder = {
    0: 'apple',
    1: 'banana',
    2: 'blackgram',
    3: 'chickpea',
    4: 'coconut',
    5: 'coffee',
    6: 'cotton',
    7: 'grapes',
    8: 'jute',
    9: 'kidneybeans',
    10: 'lentil',
    11: 'maize',
    12: 'mango',
    13: 'mothbeans',
    14: 'mungbean',
    15: 'muskmelon',
    16: 'orange',
    17: 'papaya',
    18: 'pigeonpeas',
    19: 'pomegranate',
    20: 'rice',
    21: 'watermelon'
}

# Define the function to get recommendations
def get_recommendations(model, input_data):
    encoded_prediction = model.predict(input_data)
    decoded_prediction = [label_decoder[label] for label in encoded_prediction]
    return decoded_prediction

# Function to predict from CSV
def predict_from_csv(file, model):
    data = pd.read_csv(file)
    predictions = model.predict(data)
    data['Crop_Recommendation'] = predictions
    return data

# Load the trained model
model = load_model('model/clf_final.pkl')

# Streamlit app
st.title('Crop Recommendation App')

# Sidebar options
st.sidebar.header('Options')
option = st.sidebar.selectbox('Select an option:', ('Predict using CSV', 'Manual input'))

# CSV Upload Option
if option == 'Predict using CSV':
    st.header('Upload a CSV file')
    file = st.file_uploader('Upload CSV', type=['csv'])

    if file is not None:
        predictions_df = predict_from_csv(file, model)
        st.write(predictions_df)

        # Option to download the predictions
        csv = predictions_df.to_csv(index=False)
        st.download_button('Download Predictions', csv, 'predictions.csv', 'text/csv')

# Manual Input Option
elif option == 'Manual input':
    st.header('Input Crop Features')

    # Input fields for manual input
    nitrogen = st.number_input('Nitrogen (N) in kg/ha', min_value=0.0)
    phosphorus = st.number_input('Phosphorus (P) in kg/ha', min_value=0.0)
    potassium = st.number_input('Potassium (K) in kg/ha', min_value=0.0)
    temperature = st.number_input('Temperature in Celsius', min_value=0.0)
    humidity = st.number_input('Humidity in percentage', min_value=0.0)
    ph = st.number_input('pH value of the soil', min_value=0.0)
    rainfall = st.number_input('Rainfall in mm', min_value=0.0)

    input_data = prepare_input_data(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)

    # Button to predict
    if st.button('Recommend Crop'):
        recommendations = get_recommendations(model, input_data)
        result_text = recommendations[0]  # Assuming the model returns crop name or label
        st.markdown(f"### Recommended Crop: **{result_text}**")

