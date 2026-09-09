# Compass — AI Career Path & Skill-Gap Predictor

**Student:** Rasikh Haleem
**Domain:** AI & ML

## What this project does

Compass is a machine learning web application built for AI/ML students to
plan their specialization. A student fills a short form with their academic
scores, self-rated technical skills, project/internship history, and stated
interest area. The trained model then:

1. **Predicts the best-fit career path** out of five tracks: Data Science,
   AI/ML Engineering, Web Development, Cyber Security, and Cloud/DevOps
   Engineering.
2. **Shows a confidence ranking** across all five tracks, not just the top
   pick, so the student can see close alternatives.
3. **Generates a personalized skill-gap report** — the specific skills that
   fall short of the target level for the recommended track, with the
   current value and the target value shown side by side.

This combination of prediction *and* an actionable gap report is what makes
the project different from a typical single-output classifier: the output
is not just a label, it is a plan.

## Why this is a good fit for AI & ML

- Uses supervised learning (Random Forest classification) with a real
  train/test evaluation, not just a hardcoded rule set.
- Includes feature engineering (one-hot encoding of the interest area) and
  a label encoder, following standard ML pipeline practice.
- The skill-gap module layers a rule-based explanation system on top of the
  model output, showing applied AI beyond a bare prediction.
- Fully working end-to-end: dataset generation, training, evaluation, and a
  deployed web interface.

## Project structure

```
career-predictor/
├── app.py                     Flask application (backend + routes)
├── train_model.py             Trains the Random Forest model
├── requirements.txt           Python dependencies
├── data/
│   ├── generate_dataset.py    Creates the synthetic training dataset
│   └── student_career_data.csv
├── model/
│   ├── career_model.pkl       Trained model
│   ├── label_encoder.pkl      Encodes career path labels
│   └── feature_columns.json   Feature order used at prediction time
├── templates/
│   └── index.html             Form and results page
└── static/
    └── style.css               Page styling
```

## How to run it

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. (Optional — a trained model is already included) Regenerate the dataset
   and retrain the model:
   ```
   python data/generate_dataset.py
   python train_model.py
   ```
3. Start the web application:
   ```
   python app.py
   ```
4. Open a browser at `http://127.0.0.1:5000`.

## Model performance

The Random Forest classifier reaches about **70% accuracy** on a held-out
test set across five career-path classes (a random guess would score about
20%). Data Science, AI/ML Engineering, and Web Development are predicted
most reliably; Cloud/DevOps and Cyber Security are harder to separate
because they share more overlapping skill patterns in the training data —
this is noted here honestly rather than hidden, and is a good discussion
point during a project defense.

## Possible extensions

- Replace the synthetic dataset with real, anonymized student records.
- Add a second model (e.g. gradient boosting) and compare accuracy.
- Turn the skill-gap report into a suggested learning roadmap with linked
  courses for each missing skill.
