import numpy as np #linear algebra 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#load the dataset 
df = pd.read_csv("data/loan_data.csv")

#know the shape of the data (rows, columns)
df.shape

#quick view on inside the data 
df.head(5)

#know the complete dataset information 
df.info()

#Statistical summary of all the numerical columns
df.describe()

#check duplicates
df.duplicated().sum()

#check if the data has null values 
df.isnull().sum()


#There is no null and duplicates in the data , moving forward with dtypes of the columns 

#nominal
#person_gender is nominal column
#person_home_ownership is nominal column
#loan_intent is a nominal column

#ordinal

#previous_loan_defaults_on_file is ordinal
#person_eductaion is ordinal column


# 3 nonimal and 2 ordinal

#Feature Engineering => converting categorical text values

#onhotencoding on nominal column
#onhotendcoing convert categorical data to numrical data type - pd.get_dummies    <>   int64  
#ordinalencoding for ordinal

#Onehotencoding on nominal columns 
df = pd.get_dummies(df, columns=['person_gender', 'person_home_ownership', 'loan_intent'],dtype= "int64", drop_first=True)
#dtype helps to change the data type
#drop_first helps to delete first column
#df = will assign the new values inplace of older values


#left with 2 categorical data
#person_education and previous_loan_defaults_on_file
#Ordinal columns

#ordinalencoding on ordinal columns
df['person_education']=df['person_education'].map({'Doctorate':5,'Master':4,'Bachelor':3,'High School':2,'Associate':1})

#ordinalencoding on ordinal columns
df['previous_loan_defaults_on_file']=df['previous_loan_defaults_on_file'].map({'Yes' :1, 'No': 0})

#anotherway
#d = {'Doctorate':5,'Master':4,'Bachelor':3,'High School':2,'Associate':1}
##ordinalencoding on ordinal columns
#df['person_education']=df['person_education'].map(d)


df.info() #check the dtypes 
df.head() #quick view 

#THE DATA IS READY TO GET TRAINED '

#Separate independent and dependent variables
X= df.drop('loan_status', axis=1) #Features
y= df['loan_status'] #Target variable

#training the model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#test size : split dataset into training and testing sets (80% train, 20% test)
#random state controls randomness in your model or data split

X_train.shape

X_test.shape

y_train

y_test


#Training (X_train, y_train)
#Testing (X_test, y_test) , it will give y_pred

# Importing multiple classification models to compare performance.

# Decision Tree : Single tree-based classifier
# Random Forest : Ensemble of multiple decision trees
# XGBoost : Gradient boosting algorithm (high performance)
# Logistic Regression : Linear classification algorithm (baseline model)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Goal: Training all the above models individually to evaluate and identify the best performing model for Loan Prediction.

#Initializing the all the models 

dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
xgb = XGBClassifier()
lr = LogisticRegression()

#Train each model using the training dataset

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
lr.fit(X_train, y_train)


#Testing
#Performing inference on the test set to compare predictive performance

y_dtpred = dt.predict(X_test)
y_rfpred = rf.predict(X_test)
y_xgbpred = xgb.predict(X_test)
y_lrpred = lr.predict(X_test)


#Evaluation metics
#calculate accuracy, precision,recall,f1
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


#Decision Tree Performance
print(" DT Accuracy:", accuracy_score(y_test, y_dtpred))
print("DT Precision:", precision_score(y_test, y_dtpred))
print("DT Recall:", recall_score(y_test, y_dtpred))
print("DT F1 Score:", f1_score(y_test, y_dtpred))


#Random Forest Performance
print("RF Accuracy:", accuracy_score(y_test, y_rfpred))
print("RF Precision:", precision_score(y_test, y_rfpred))
print("RFRecall:", recall_score(y_test, y_rfpred))
print("RF F1 Score:", f1_score(y_test, y_rfpred))

#XGBoost Performance
print("XGB Accuracy:", accuracy_score(y_test, y_xgbpred))
print("XGB Precision:", precision_score(y_test, y_xgbpred))
print("XGB Recall:", recall_score(y_test, y_xgbpred))
print("XGB F1 Score:", f1_score(y_test, y_xgbpred))

#Logistic Regression Performance
print("LR Accuracy:", accuracy_score(y_test, y_lrpred))
print("LR Precision:", precision_score(y_test, y_lrpred))
print("LR Recall:", recall_score(y_test, y_lrpred))
print("LR F1 Score:", f1_score(y_test, y_lrpred))



#Loading dependencies for model persistence and directory management using joblib and os 
import joblib
import os

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Since XGB is best model
best_model = xgb

# Retrain on full dataset
best_model.fit(X, y)

# Save model
joblib.dump(best_model, "models/model.pkl")

# Save column order
joblib.dump(X.columns.tolist(), "models/columns.pkl")

print("Model and columns saved successfully!")

