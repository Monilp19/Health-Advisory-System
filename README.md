# Health-Advisory-System
It is a Project based on ML and Python, in this project we have used several datasets in form of CSV through which we can train our model, after training using Random Forest Classifier an app is created using streamlit python which will be later replaced by a better frontend and there are way more functions to add in this project.
The Streamlit application checks all four datasets every 2 seconds:

- `data/disease_symptom.csv` — ML training data
- `data/symptoms_description.csv` — patient-friendly disease descriptions
- `data/precautions.csv` — precautions/reference guidance
- `data/symptom_severity.csv` — symptom severity weights

You can edit and save any CSV while Streamlit is running.

### What happens after an edit?

- Edit `disease_symptom.csv` → the ML model automatically retrains.
- Edit `symptom_severity.csv` → the model automatically retrains and the symptom severity calculation updates.
- Edit `symptoms_description.csv` → the displayed description updates.
- Edit `precautions.csv` → the displayed precautions update.
- Add/remove symptom rows or disease rows → the available data is detected on the next check.

No manual restart is required.

## Run

Type in Terminal :-

pip install -r requirements.txt
then...
streamlit run app.py


## Optional manual model training

Type in Terminal:- 

python train_model.py


The application also trains automatically from the current CSV files, so manual training is not required for normal use.

## Important

This is an educational machine-learning demonstration. Predictions are not medical diagnoses and should not be used to make treatment decisions.
