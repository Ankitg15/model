import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Make sure to download the necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

# Load your dataset
data = pd.read_csv('emails.csv')

# Data preprocessing function
def preprocess_text(text):
    # Convert text to lower case
    text = text.lower()
    # Remove HTML tags
    pattern = re.compile('<.*?>')
    text = pattern.sub('', text)
    # Tokenize the text using NLTK's word_tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.lower() not in stop_words and token not in string.punctuation]
    return ' '.join(tokens)  # Join tokens back to a single string

# Apply preprocessing to the "Text" column
data['Tokenized_Text'] = data['Text'].apply(preprocess_text)

# Save preprocessed data (if needed)
data.to_csv('preprocessed_dataset.csv', index=False)

# Load preprocessed data
preprocessed_data = pd.read_csv('preprocessed_dataset.csv')
# print(preprocessed_data.head())

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(preprocessed_data['Tokenized_Text'])

# Labels
y = preprocessed_data['Spam']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.2f}')

# Optionally, print more detailed metrics
# print(classification_report(y_test, y_pred))

# Function to classify a new email
def classify_email(email):
    # Preprocess the email
    # processed_email = preprocess_text(email)
    # Vectorize the email
    email = vectorizer.transform([email])
    # Predict using the trained classifier
    prediction = classifier.predict(email)
    return 'Phishing' if prediction == 1 else 'Legitimate'

# Take user input and classify
user_input = input("Enter an email text to classify: ")
result = classify_email(user_input)
print(f'The email is classified as: {result}')
