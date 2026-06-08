import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

print("1. Loading datasets (this might take a minute)...")
train_df = pd.read_csv('fraudTrain.csv')
test_df = pd.read_csv('fraudTest.csv')

print("\n2. Balancing the Training Data...")
# Separate the fraud and legitimate transactions
legit = train_df[train_df.is_fraud == 0]
fraud = train_df[train_df.is_fraud == 1]

# Randomly sample the legitimate transactions to match the number of fraud transactions
legit_sampled = legit.sample(n=len(fraud), random_state=42)

# Combine and shuffle the data
balanced_df = pd.concat([legit_sampled, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Balanced Dataset Size: {len(balanced_df)} rows")

print("\n3. Selecting Features...")
# To keep the model fast and accurate, we select key numerical columns 
# (transaction amount, locations, population, and time)
features = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']

# Set up our Training data (from our new balanced dataset)
X_train = balanced_df[features]
y_train = balanced_df['is_fraud']

# Set up our Testing data (from the completely unseen test_df)
X_test = test_df[features]
y_test = test_df['is_fraud']

print("\n4. Training the Random Forest Model...")
# Initialize the model (n_jobs=-1 tells your computer to use all its processing power)
model = RandomForestClassifier(random_state=42, n_jobs=-1)
# Teach the model using our balanced training data
model.fit(X_train, y_train)

print("\n5. Testing the Model...")
# Ask the model to predict fraud on the unseen test dataset
y_pred = model.predict(X_test)

print("\n--- Final Model Evaluation ---")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nTask 2 Code Complete!")
