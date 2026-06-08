import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print("1. Loading Customer Churn Dataset...")

# The script is looking for the exact file name we just fixed in your folder!
file_path = 'Churn_Modelling.csv' 

try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded! Total records: {len(df)}")
except FileNotFoundError:
    print(f"\n❌ Error: Could not find '{file_path}'.")
    print("Check your Music folder and make sure the file is named exactly: Churn_Modelling.csv")
    exit()

print("\n2. Preprocessing Data...")
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname', 'customerID']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

target_column = 'Exited' if 'Exited' in df.columns else 'Churn'

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=[target_column])
y = df[target_column]

print("\n3. Splitting Data into Train and Test Sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n4. Training the Random Forest Classifier...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\n5. Evaluating Model Performance...")
y_pred = model.predict(X_test)

print("\n--- Final Model Evaluation ---")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nTask 3 Code Complete! All 3 tasks finished!")
