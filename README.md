# Mental Health Severity Prediction

This project predicts the severity of depression and anxiety using machine learning models trained on mental health data. It employs XGBoost models to assist in mental health diagnostics by providing predictions for depression and anxiety levels.
 
---
## Links
[Kaggle](https://www.kaggle.com/code/arittrabag/mental-health-self-analysis) | [HugginFaceSpaces](https://huggingface.co/spaces/arittrabag/Mental-Health-Severity-Prediction)

## Dataset Preprocessing Steps

### Handling Missing Values
- Filled missing values using mean, median, or mode based on column characteristics.

### Feature Selection
Key features used:
- **Demographics**: Age, Gender, BMI, WHO BMI classification
- **Mental Health Scores**: PHQ Score (depression), GAD Score (anxiety), Epworth Score (sleepiness)
- **Mental Health Indicators**: Depressiveness, Suicidal thoughts, Anxiousness
- **Diagnosis & Treatment**: Depression/Anxiety Diagnosis, Depression/Anxiety Treatment

### Feature Engineering
- Binary encoding for categorical features (e.g., `Gender`, `Suicidal thoughts` → `0`/`1`).

### Normalization/Scaling
- Numerical features (BMI, PHQ Score, etc.) scaled to improve model performance.

### Training/Test Split
- 80/20 split for training and evaluation.

---

## Model Selection Rationale

### XGBoost Performance
- **Depression Severity Model**: Accuracy = `1.0000`
- **Anxiety Severity Model**: Accuracy = `0.9873`

### Why XGBoost?
- Handles large datasets and feature interactions effectively.
- Scalable, fast, and resistant to overfitting when tuned.

---

## How to Run the Inference Script

### Step 1: Setup
1. Clone the repository:
   ```
   git clone https://github.com/Arittra-Bag/Self-Analysis-Mental-Health-Model.git
   cd Self-Analysis-Mental-Health-Model
    ```
2. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Ensure you have the trained models (`xgb_model_depression.pkl` and `xgb_model_anxiety.pkl`) placed in the project directory under `Models` Folder.

### Step 2: Running the Script
You can run the inference script in the following way-

1. Using the command line:
```
python predict_mental_health.py
```
2. Parameters:

- The script takes multiple parameters like age, gender, bmi, etc.
- These parameters will be passed as input, and the script will output the predicted severity of depression and anxiety.
#### Example usage in script:
```
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

# Predicting depression and anxiety severity
depression_pred = predict_depression_severity(input_data)
anxiety_pred = predict_anxiety_severity(input_data)

print(f"Predicted Depression Severity: {depression_pred}")
print(f"Predicted Anxiety Severity: {anxiety_pred}")
```
## UI Usage Instructions

#### Using Gradio Interface-

- A simple Gradio interface has been implemented to visualize the prediction process.

- To launch the interface, run the following command:
```
python mental_health_ui.py
```

- You will be able to input your mental health data (such as age, gender, scores, etc.) and get predictions for depression and anxiety severity.

- The interface allows you to submit inputs and get predictions without writing code.

## Conclusion
This project helps in predicting depression and anxiety severity using machine learning models. The XGBoost models provide reliable and quick predictions, and the inclusion of model interpretability methods (like SHAP) ensures transparency and better understanding of the results.
