def load_model(model_path):
    import joblib
    model = joblib.load(model_path)
    return model

def prepare_input_data(input_data):
    import pandas as pd
    # Assuming input_data is a dictionary, convert it to a DataFrame
    df = pd.DataFrame([input_data])
    return df

def get_recommendations(model, input_data):
    prepared_data = prepare_input_data(input_data)
    predictions = model.predict(prepared_data)
    return predictions.tolist()