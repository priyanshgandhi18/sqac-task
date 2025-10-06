import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Create the dataset
data = {
    'text': [
        "I am so happy and excited for the party tonight!",
        "I can't believe we won the game, this is amazing!",
        "I had a wonderful time with my friends today.",
        "I'm feeling really down and upset about what happened.",
        "It's been a tough day, I just want to be alone.",
        "I miss the old days, everything feels so bleak now.",
        "The meeting is scheduled for 3 PM in the conference room.",
        "Please remember to pick up milk on your way home.",
        "The sky is blue and the sun is shining.",
        "I have a million things to do and not enough time.",
        "The pressure at work is becoming unbearable.",
        "I'm so overwhelmed with all these deadlines."
    ],
    'mood': ['happy', 'happy', 'happy', 'sad', 'sad', 'sad', 'neutral', 'neutral', 'neutral', 'stressed', 'stressed', 'stressed']
}
df = pd.DataFrame(data)

# Define features (X) and target (y)
X = df['text']
y = df['mood']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Fit and transform the training data, and transform the test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# --- Train and Evaluate Logistic Regression Model ---
print("--- Logistic Regression Model ---")
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, lr_predictions))
print("\nClassification Report:\n", classification_report(y_test, lr_predictions))


# --- Train and Evaluate Naive Bayes Model ---
print("\n--- Naive Bayes Model ---")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Make predictions
nb_predictions = nb_model.predict(X_test_tfidf)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print("\nClassification Report:\n", classification_report(y_test, nb_predictions))

#### 4. Conclusion

