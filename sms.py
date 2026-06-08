import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("1. Loading SMS Dataset...")
# We use encoding='latin-1' because text files often contain special characters
df = pd.read_csv('spam.csv', encoding='latin-1')

# Clean up the dataset (Kaggle versions often have unnamed blank columns)
df = df.dropna(how="any", axis=1)
df.columns = ['label', 'message']

print(f"Dataset loaded: {len(df)} messages found.")
print("\nClass Distribution (ham = legitimate, spam = junk):")
print(df['label'].value_counts())

print("\n2. Splitting Data into Train and Test Sets...")
# X contains the raw text messages, y contains the labels ('ham' or 'spam')
X = df['message']
y = df['label']

# Split: 80% for training the model, 20% for testing its accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n3. Converting Text to Numbers using TF-IDF...")
# Vectorizer converts words into numerical features based on importance
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

# Train the vectorizer on training text and transform it
X_train_tfidf = vectorizer.fit_transform(X_train)
# Transform the testing text using the same rules
X_test_tfidf = vectorizer.transform(X_test)

print("\n4. Training Multinomial Naive Bayes Classifier...")
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

print("\n5. Evaluating Model Performance...")
y_pred = classifier.predict(X_test_tfidf)

print("\n--- Final Model Evaluation ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n6. Testing Custom Examples:")
# Let's see how it handles brand new text sentences
custom_messages = [
    "Hey, are we still meeting up for badminton practice tonight?",
    "CONGRATULATIONS! You have won a 1,000 cash prize. Call 09061701461 to claim now. Urgent!"
]

custom_tfidf = vectorizer.transform(custom_messages)
predictions = classifier.predict(custom_tfidf)

for msg, prediction in zip(custom_messages, predictions):
    print(f"\nMessage: '{msg}'\nPrediction: {prediction.upper()}")

print("\nTask 4 Code Complete!")
