import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier as xgb
from sklearn.metrics import accuracy_score

data = pd.read_csv("ad_10000records.csv")
print(data.head())

print(data["Clicked on Ad"].value_counts())

# Click through rate
click_through_rate = 4917 / 10000 * 100
print(f"The click through rate is: {click_through_rate}%")

le = LabelEncoder()

data["Gender"] = le.fit_transform(data["Gender"])

# Taking the first seven columns of the dataset as features
x=data.iloc[:,0:7]
# Dropping the Categorical Variables
x=x.drop(['Ad Topic Line','City'],axis=1)

# Assigning the final variable as the target variable
y=data.iloc[:,9]
# Using train test split to split the dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# XGBoostClassifier model
model = xgb(random_state=42)

# Training the model
model.fit(x_train, y_train)

# Predicting test dataset values with the model
y_pred = model.predict(x_test)

# Accuracy check of the model prediction
print("The model accuracy is", accuracy_score(y_test,y_pred))