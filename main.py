import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


data = pd.read_csv('churn.csv')


data.drop('customerID', axis=1, inplace=True)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(inplace=True)
data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
label_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
              'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
              'PaperlessBilling', 'PaymentMethod']
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])


features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
X = data[features]
y = data['Churn']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [50, 100, 200]
}

grid_search = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Model evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Best Hyperparameters:', grid_search.best_params_)
print('Accuracy:', accuracy)
print('Classification Report:')
print(classification_report(y_test, y_pred))
