from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data
raw_mail_data = pd.read_csv('mail_data.csv')
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1
X = mail_data["Message"]
Y = mail_data["Category"].astype('int')

# Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# TF-IDF Feature Extraction
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Logistic Regression Model
lg_model = LogisticRegression()
lg_model.fit(X_train_features, Y_train)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_features, Y_train)

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_features, Y_train)

# Support Vector Machine Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_features, Y_train)

# Set the best model
best_model = svm_model

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('ui.html')

@app.route('/predict_spam', methods=['POST'])
def predict_spam():
    try:
        data = request.json
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Transform the input text using TF-IDF
        transformed_data = feature_extraction.transform([text]).toarray()
        
        # Predict using the best model
        predicted_spam = best_model.predict(transformed_data)[0]
        prediction = 'Ham' if predicted_spam == 1 else 'Spam'
        
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
