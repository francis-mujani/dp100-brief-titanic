import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score


train_data = pd.read_csv("titanic.csv")

features = ["Age","Pclass","SibSp", "Parch", "Fare","Sex", "Embarked"]

X = pd.get_dummies(train_data[features])
y = train_data["Survived"]

# missing values
imputer = SimpleImputer(strategy='most_frequent')
imputed_X = pd.DataFrame(imputer.fit_transform(X))
imputed_X.columns = X.columns
imputed_X[["Age","Pclass","SibSp", "Parch", "Fare"]] = imputed_X[["Age","Pclass","SibSp", "Parch", "Fare"]].astype('int')


# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(imputed_X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y_train)


# score
predictions = model.predict(X_valid)

mae = mean_absolute_error(predictions.astype('int'), y_valid)
acc = accuracy_score(y_valid, predictions.astype('int'))
print("mae : {}, accuracy : {}".format(mae, acc))

# Save the trained model in the outputs folder
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/titanic_model.pkl')

