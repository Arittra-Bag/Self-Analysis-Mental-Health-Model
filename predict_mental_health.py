import pandas as pd
import joblib

# Load the trained model
xgb_model_depression = joblib.load('Models/xgb_model_depression.pkl')
xgb_model_anxiety = joblib.load('Models/xgb_model_anxiety.pkl')

# Depression and Anxiety Severity Class Mappings
depression_classes = {
    0: "No Depression",
    1: "Mild Depression",
    2: "Moderate Depression",
    3: "Severe Depression",
    4: "Very Severe Depression",
    5: "Extreme Depression"
}

anxiety_classes = {
    0: "No Anxiety",
    1: "Mild Anxiety",
    2: "Moderate Anxiety",
    3: "Severe Anxiety",
    4: "Extreme Anxiety"
}

# Function to predict depression severity
def predict_depression_severity(input_data):
    # Ensure the input is a DataFrame
    input_df = pd.DataFrame([input_data])

    # Remove bmi_category if it's in the input
    input_df = input_df.drop(columns=['bmi_category'], errors='ignore')

    # Predict depression severity
    prediction = xgb_model_depression.predict(input_df)
    return depression_classes.get(prediction[0], "Unknown Depression Severity")

# Function to predict anxiety severity
def predict_anxiety_severity(input_data):
    # Ensure the input is a DataFrame
    input_df = pd.DataFrame([input_data])

    # Remove bmi_category if it's in the input
    input_df = input_df.drop(columns=['bmi_category'], errors='ignore')

    # Predict anxiety severity
    prediction = xgb_model_anxiety.predict(input_df)
    return anxiety_classes.get(prediction[0], "Unknown Anxiety Severity")

# Example input data (without bmi_category)
input_data = {
    'age': 19,
    'gender': 1,
    'bmi': 2.17,
    'who_bmi': 0,
    'phq_score': 0.42,
    'depressiveness': 0,
    'suicidal': 0,
    'depression_diagnosis': 0,
    'depression_treatment': 0,
    'gad_score': 0.88,
    'anxiousness': 2,
    'anxiety_diagnosis': 1,
    'anxiety_treatment': 0,
    'epworth_score': 0.16,
    'sleepiness': 0
}

# Predict depression and anxiety severity
depression_pred = predict_depression_severity(input_data)
anxiety_pred = predict_anxiety_severity(input_data)

# Output the prediction
print(f"Predicted Depression Severity: {depression_pred}")
print(f"Predicted Anxiety Severity: {anxiety_pred}")