from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import numpy as np

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('online_shoppers_intention.csv')

# --- Data Preprocessing ---
# Convert boolean columns to integers
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)

# Separate features (X) and target (y)
X = df.drop('Revenue', axis=1)
y = df['Revenue']

# Identify categorical and numerical features
categorical_features = ['Month', 'VisitorType']
numerical_features = ['Administrative', 'Administrative_Duration', 'Informational', 
                     'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
                     'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'OperatingSystems',
                     'Browser', 'Region', 'TrafficType', 'Weekend']

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- Model Training ---
# Create pipelines for different models
models = {
    'lr': Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', LogisticRegression(random_state=42, max_iter=1000))]),
    'dt': Pipeline(steps=[('preprocessor', preprocessor),
                         ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5))]),
    'knn': Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', KNeighborsClassifier(n_neighbors=7))]),
    'ens': Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', VotingClassifier(estimators=[
                              ('lr', LogisticRegression(random_state=42, max_iter=1000)),
                              ('dt', DecisionTreeClassifier(random_state=42, max_depth=5)),
                              ('knn', KNeighborsClassifier(n_neighbors=7))
                          ], voting='soft'))])
}

# Train all models
for model in models.values():
    model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.form.to_dict()

    # Get the selected algorithm
    algorithm = data.get('algorithm', 'lr')  # default to lr

    try:
        # Map numerical categorical values back to dataset strings
        month_map = {'2': 'Feb', '5': 'Mar', '6': 'May', '4': 'June', '3': 'Jul', 
                     '0': 'Aug', '9': 'Sep', '8': 'Oct', '7': 'Nov', '1': 'Dec'}
        visitor_map = {'1': 'New_Visitor', '2': 'Returning_Visitor', '0': 'Other'}
        
        # Handle Boolean/Integer conversion safely
        weekend_val = data.get('Weekend', 0)
        is_weekend = int(weekend_val) if str(weekend_val).isdigit() else 0
    
        # Convert data to a pandas DataFrame
        new_data = pd.DataFrame({
            'Administrative': [int(data['Administrative'])],
            'Administrative_Duration': [float(data['Administrative_Duration'])],
            'Informational': [int(data['Informational'])],
            'Informational_Duration': [float(data['Informational_Duration'])],
            'ProductRelated': [int(data['ProductRelated'])],
            'ProductRelated_Duration': [float(data['ProductRelated_Duration'])],
            'BounceRates': [float(data['BounceRates'])],
            'ExitRates': [float(data['ExitRates'])],
            'PageValues': [float(data['PageValues'])],
            'SpecialDay': [float(data['SpecialDay'])],
            'Month': [month_map.get(str(data.get('Month')), 'May')],
            'OperatingSystems': [int(data['OperatingSystems'])],
            'Browser': [int(data['Browser'])],
            'Region': [int(data['Region'])],
            'TrafficType': [int(data['TrafficType'])],
            'VisitorType': [visitor_map.get(str(data.get('VisitorType')), 'Returning_Visitor')],
            'Weekend': [int(is_weekend)] # Cast to int for consistency with training data
        })
    except (ValueError, KeyError) as e:
        return render_template('index.html', prediction_text=f"Error in input data: {str(e)}")

    # Get the selected model
    model = models.get(algorithm)
    if not model:
        return render_template('index.html', prediction_text="Invalid algorithm selected")

    # Make prediction
    prediction = model.predict(new_data)
    probability = model.predict_proba(new_data)

    if prediction[0] == 1:
        result = "The model predicts that the user WILL generate revenue."
    else:
        result = "The model predicts that the user will NOT generate revenue."
    
    prob_text = f"Probability of not generating revenue: {probability[0][0]:.4f}, Probability of generating revenue: {probability[0][1]:.4f}"


    return render_template('index.html', prediction_text=result, probability_text=prob_text)

if __name__ == "__main__":
    app.run(debug=True)
