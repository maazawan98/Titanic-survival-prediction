import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv") 


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical variables
label_encoder_sex = LabelEncoder()
label_encoder_embarked = LabelEncoder()

df['Sex'] = label_encoder_sex.fit_transform(df['Sex'])
df['Embarked'] = label_encoder_embarked.fit_transform(df['Embarked'])
# Features and target
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Enter passenger details to predict survival:")
pclass = int(input("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd): "))
sex_input = input("Sex (male/female): ").strip().lower()
age = float(input("Age: "))
sibsp = int(input("Number of siblings/spouses aboard: "))
parch = int(input("Number of parents/children aboard: "))
fare = float(input("Passenger Fare: "))
embarked_input = input("Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton): ").strip().upper()

sex_encoded = label_encoder_sex.transform([sex_input])[0]
embarked_encoded = label_encoder_embarked.transform([embarked_input])[0]

user_data = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex_encoded,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked_encoded
}])

# Predict
prediction = model.predict(user_data)[0]
probability = model.predict_proba(user_data)[0][1]

# Output result
if prediction == 1:
    print(f"\nThe passenger would have SURVIVED (Probability: {probability:.2%})")
else:
    print(f"\n The passenger would NOT have survived (Probability: {probability:.2%})")
