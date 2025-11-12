import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

DATA_PATH = 'indian_startup_funding.csv'

def encode_with_unknown(series):
    categories = series.unique().tolist()
    if 'unknown' not in categories:
        categories.append('unknown')
    return pd.Categorical(series, categories=categories)

def train_and_save_model():
    data = pd.read_csv(DATA_PATH)

    median_funding = data['Funding Amount in $'].median()
    data['Success'] = np.where(data['Funding Amount in $'] > median_funding, 1, 0)

    features = ['City', 'Starting Year', 'Industries', 'No. of Employees', 
                'Funding Round', 'No. of Investors', 'Funding Amount in $']
    X = data[features].copy()
    y = data['Success']

    X['City'] = encode_with_unknown(X['City'])
    X['Industries'] = encode_with_unknown(X['Industries'])
    X['No. of Employees'] = encode_with_unknown(X['No. of Employees'])

    X['City_enc'] = X['City'].cat.codes
    X['Industries_enc'] = X['Industries'].cat.codes
    X['Employees_enc'] = X['No. of Employees'].cat.codes

    X_model = X[['City_enc', 'Starting Year', 'Industries_enc', 'Employees_enc', 
                 'Funding Round', 'No. of Investors', 'Funding Amount in $']]

    X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification report on test data:")
    print(classification_report(y_test, y_pred))

    # Save model and categories to files
    joblib.dump(model, 'startup_success_model.pkl')
    joblib.dump(X['City'].cat.categories, 'city_categories.pkl')
    joblib.dump(X['Industries'].cat.categories, 'industries_categories.pkl')
    joblib.dump(X['No. of Employees'].cat.categories, 'employees_categories.pkl')

if __name__ == '__main__':
    train_and_save_model()
