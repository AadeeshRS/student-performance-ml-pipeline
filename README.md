# Student Performance ML Project

An end-to-end machine learning project that predicts students' math scores from demographic and academic features. The project includes data ingestion, preprocessing, model training with multiple regressors, and a Flask web app for inference.

## Features
- Data ingestion from a CSV dataset and train/test split
- Preprocessing pipeline with imputation, scaling, and one-hot encoding
- Model selection across several regressors and basic hyperparameter search
- Model and preprocessor saved as artifacts for inference
- Flask UI for single prediction via a web form

## Project Structure
```
app.py                     # Flask entry point (same as application.py)
application.py             # Flask entry point
artifact/                  # Generated artifacts (train/test data, model, preprocessor)
notebook/                  # EDA and training notebooks + dataset source
src/
	components/              # Data ingestion, transformation, model training
	pipeline/                # Training and prediction pipeline
templates/                 # Flask HTML templates
```

## Dataset
The training data is loaded from:
```
notebook/Data/StudentsPerformance.csv
```
Target: `math score`

## Setup
1. Create and activate a Python environment (3.8+ recommended).
2. Install dependencies:
```
pip install -r requirements.txt
```

## Train the Model
This pipeline reads the dataset, builds the preprocessing pipeline, trains multiple regressors, and saves the best model.
```
python src/components/data_ingestion.py
```
Artifacts are written to:
```
artifact/model.pkl
artifact/preprocessor.pkl
```

## Run the Web App
Start the Flask server:
```
python application.py
```
Then open:
```
http://localhost:5000
```

## Inference Inputs
The form collects:
- `gender`
- `race/ethnicity`
- `parental level of education`
- `lunch`
- `test preparation course`
- `reading score`
- `writing score`

The app returns the predicted math score.

## Notes
- `application.py` and `app.py` are equivalent; use either as the app entry point.
- `src/pipeline/train_pipeline.py` is currently empty; training is invoked via `data_ingestion.py`.

## License
This project is for educational use.