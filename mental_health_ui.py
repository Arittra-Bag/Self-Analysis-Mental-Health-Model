import gradio as gr
import pandas as pd
import joblib

# Load the trained models
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

# Prediction functions
def predict_depression_severity(age, gender, bmi, who_bmi, phq_score, depressiveness, suicidal, depression_diagnosis, depression_treatment, gad_score, anxiousness, anxiety_diagnosis, anxiety_treatment, epworth_score, sleepiness):
    # Convert categorical values to numeric
    gender = 0 if gender == "Male" else 1
    suicidal = int(suicidal)  # Convert to int (0 or 1)
    depression_diagnosis = int(depression_diagnosis)  # Convert to int (0 or 1)
    depression_treatment = int(depression_treatment)  # Convert to int (0 or 1)
    anxiety_diagnosis = int(anxiety_diagnosis)  # Convert to int (0 or 1)
    anxiety_treatment = int(anxiety_treatment)  # Convert to int (0 or 1)
    sleepiness = int(sleepiness)  # Convert to int (0 or 1)

    input_data = {
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'who_bmi': who_bmi,
        'phq_score': phq_score,
        'depressiveness': depressiveness,
        'suicidal': suicidal,
        'depression_diagnosis': depression_diagnosis,
        'depression_treatment': depression_treatment,
        'gad_score': gad_score,
        'anxiousness': anxiousness,
        'anxiety_diagnosis': anxiety_diagnosis,
        'anxiety_treatment': anxiety_treatment,
        'epworth_score': epworth_score,
        'sleepiness': sleepiness
    }
    
    # Ensure the input is a DataFrame
    input_df = pd.DataFrame([input_data])
    prediction = xgb_model_depression.predict(input_df)
    
    # Map prediction to readable class
    return depression_classes[prediction[0]]

def predict_anxiety_severity(age, gender, bmi, who_bmi, phq_score, depressiveness, suicidal, depression_diagnosis, depression_treatment, gad_score, anxiousness, anxiety_diagnosis, anxiety_treatment, epworth_score, sleepiness):
    # Convert categorical values to numeric
    gender = 0 if gender == "Male" else 1
    suicidal = int(suicidal)  # Convert to int (0 or 1)
    depression_diagnosis = int(depression_diagnosis)  # Convert to int (0 or 1)
    depression_treatment = int(depression_treatment)  # Convert to int (0 or 1)
    anxiety_diagnosis = int(anxiety_diagnosis)  # Convert to int (0 or 1)
    anxiety_treatment = int(anxiety_treatment)  # Convert to int (0 or 1)
    sleepiness = int(sleepiness)  # Convert to int (0 or 1)

    input_data = {
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'who_bmi': who_bmi,
        'phq_score': phq_score,
        'depressiveness': depressiveness,
        'suicidal': suicidal,
        'depression_diagnosis': depression_diagnosis,
        'depression_treatment': depression_treatment,
        'gad_score': gad_score,
        'anxiousness': anxiousness,
        'anxiety_diagnosis': anxiety_diagnosis,
        'anxiety_treatment': anxiety_treatment,
        'epworth_score': epworth_score,
        'sleepiness': sleepiness
    }
    
    # Ensure the input is a DataFrame
    input_df = pd.DataFrame([input_data])
    prediction = xgb_model_anxiety.predict(input_df)
    
    # Map prediction to readable class
    return anxiety_classes[prediction[0]]

# Wrapper function to call both predictions
def predict_both(age, gender, bmi, who_bmi, phq_score, depressiveness, suicidal, depression_diagnosis, depression_treatment, gad_score, anxiousness, anxiety_diagnosis, anxiety_treatment, epworth_score, sleepiness):
    depression_prediction = predict_depression_severity(age, gender, bmi, who_bmi, phq_score, depressiveness, suicidal, depression_diagnosis, depression_treatment, gad_score, anxiousness, anxiety_diagnosis, anxiety_treatment, epworth_score, sleepiness)
    anxiety_prediction = predict_anxiety_severity(age, gender, bmi, who_bmi, phq_score, depressiveness, suicidal, depression_diagnosis, depression_treatment, gad_score, anxiousness, anxiety_diagnosis, anxiety_treatment, epworth_score, sleepiness)
    
    return depression_prediction, anxiety_prediction

# Gradio interface setup
inputs = [
    gr.Number(label="Age (in years)", info="Enter your age."),
    gr.Radio(["Male", "Female"], label="Gender", info="Select your gender."),
    gr.Number(label="BMI", info="Enter your BMI value."),
    gr.Number(label="WHO BMI Classification", info="Enter your WHO BMI classification (0: Underweight, 1: Normal, 2: Overweight, 3: Obese)."),
    gr.Number(label="PHQ Score", info="Enter your PHQ-9 score for depression (0-27)."),
    gr.Number(label="Depressiveness", info="Enter your level of depressiveness (0-1)."),
    gr.Radio([0, 1], label="Suicidal", info="Indicate if you've experienced suicidal thoughts (0: No, 1: Yes)."),
    gr.Radio([0, 1], label="Depression Diagnosis", info="Indicate if you've been diagnosed with depression (0: No, 1: Yes)."),
    gr.Radio([0, 1], label="Depression Treatment", info="Indicate if you're receiving treatment for depression (0: No, 1: Yes)."),
    gr.Number(label="GAD Score", info="Enter your GAD-7 score for anxiety (0-21)."),
    gr.Number(label="Anxiousness", info="Enter your level of anxiousness (0-1)."),
    gr.Radio([0, 1], label="Anxiety Diagnosis", info="Indicate if you've been diagnosed with anxiety (0: No, 1: Yes)."),
    gr.Radio([0, 1], label="Anxiety Treatment", info="Indicate if you're receiving treatment for anxiety (0: No, 1: Yes)."),
    gr.Number(label="Epworth Score", info="Enter your Epworth Sleepiness Scale score (0-24)."),
    gr.Number(label="Sleepiness", info="Enter your sleepiness score (0-1).")
]

outputs = [
    gr.Textbox(label="Predicted Depression Severity", info="The predicted severity of your depression."),
    gr.Textbox(label="Predicted Anxiety Severity", info="The predicted severity of your anxiety.")
]

# Creating the interface with a submit button
interface = gr.Interface(
    fn=predict_both,
    inputs=inputs,
    outputs=outputs,
    title="Mental Health Severity Prediction",
    description="This app predicts the severity of depression and anxiety based on various inputs related to mental health. Please fill in the fields below and click submit to get predictions.",
    live=False,  # Set to False to ensure predictions are only made after submit
    allow_flagging="never",  # Prevent users from flagging the results
)

# Launch the app
interface.launch()
