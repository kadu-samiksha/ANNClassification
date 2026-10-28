# Customer Churn Prediction (ANN Classification)

A binary classification project that predicts whether a bank customer will leave (churn), using an Artificial Neural Network built with TensorFlow/Keras and served through a Streamlit web app.

## Problem Statement

Given a customer's profile (credit score, geography, age, balance, etc.), predict the probability that they will exit the bank (`Exited = 1`).

## Dataset

- **File:** `Churn_Modelling.csv` (10,000 rows)
- **Target:** `Exited` (1 = churned, 0 = retained)
- **Features used:** `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`
- **Dropped:** `RowNumber`, `CustomerId`, `Surname` (identifiers, no predictive value)

## Project Structure

```
churn-classification/
├── Churn_Modelling.csv          # Dataset
├── experiments.ipynb            # Preprocessing, ANN training, TensorBoard
├── prediction.ipynb             # Load saved model + run a sample prediction
├── app.py                       # Streamlit app
├── model.h5                     # Trained ANN model
├── label_encoder_gender.pkl     # Fitted LabelEncoder (Gender)
├── onehot_encoder_geo.pkl       # Fitted OneHotEncoder (Geography)
├── scaler.pkl                   # Fitted StandardScaler
├── requirements.txt
└── README.md
```

## Workflow

1. **Preprocessing** (`experiments.ipynb`)
   - Drop identifier columns
   - Label-encode `Gender`
   - One-hot encode `Geography`
   - 80/20 train-test split (`random_state=42`)
   - Standardize features with `StandardScaler` (fit on train only)
   - Save the encoders and scaler as `.pkl` files
2. **Model**
   - Input layer: 12 features
   - Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
   - Optimizer: Adam (lr = 0.01)
   - Loss: Binary Crossentropy
   - Callbacks: EarlyStopping (`val_loss`, patience = 10, restore best weights), TensorBoard
3. **Inference** (`prediction.ipynb`, `app.py`)
   - Apply the same saved encoders and scaler to new input, then predict
   - Probability > 0.5 → *likely to churn*, otherwise *not likely to churn*

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | _fill in from your notebook_ |
| Validation Loss | _fill in from your notebook_ |

## Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

**Run the web app**

```bash
streamlit run app.py
```

Pick the customer's details in the UI and the app shows the churn probability and a verdict.

**Retrain the model**

Open `experiments.ipynb` and run all cells. This regenerates `model.h5`, `scaler.pkl`, and both encoder `.pkl` files.

**View training curves**

```bash
tensorboard --logdir logs/fit
```

## Sample Prediction

```python
input_data = {
    'CreditScore': 600, 'Geography': 'France', 'Gender': 'Male',
    'Age': 40, 'Tenure': 3, 'Balance': 60000, 'NumOfProducts': 2,
    'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 50000
}
# Output: churn probability ≈ 0.04 → Customer is not likely to churn
```

## Tech Stack

Python · TensorFlow / Keras · scikit-learn · pandas · NumPy · Streamlit · TensorBoard

## Notes

- The scaler and encoders must come from the same training run as `model.h5`, and feature column order must match training.
- Keep this project's `scaler.pkl` separate from the regression project's. They are fit on different feature sets.

## Future Improvements

- Handle class imbalance (SMOTE / class weights)
- Add precision, recall, F1 and ROC-AUC evaluation
- Compare against tree-based baselines (XGBoost, Random Forest)
- Deploy on Streamlit Community Cloud

