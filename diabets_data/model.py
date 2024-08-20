#İmporting the Dependencies

import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm


#Data Collection and Anaylsis
diabets_data = pd.read_csv(r"C:\Users\recep\OneDrive\Masaüstü\diabets_data\diabetes.csv")
diabets_data.describe()

#0 --> non diabetic
#1 --> diabetic
diabets_data['Outcome'].value_counts()
diabets_data.groupby('Outcome').mean()

#separating the data set as X and Y 
X = diabets_data.drop(columns= "Outcome", axis= 1)
Y = diabets_data["Outcome"]

#Data Standardization

'''z= (x−μ) / σ

x: Property value to be converted

μ: Average value of the feature

σ: Standard deviation of the feature
'''

scaler = StandardScaler()
# we trasform X.values because we do not need feature names in scaling
standardized_data = scaler.fit_transform(X.values)
standardized_data
X = standardized_data
Y = diabets_data["Outcome"]

#Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size= 0.2, stratify= Y, random_state= 2)

#Training the Model
classifier = svm.SVC(kernel= 'linear')
classifier.fit(X_train, Y_train)

#Model Evaluation
#Accuracy Score on Traning Data
X_train_prediction = classifier.predict(X_train)
X_train_accuracy = accuracy_score(X_train_prediction, Y_train)
X_train_accuracy

#Accuracy Score on Test Data
X_test_prediction = classifier.predict(X_test)
X_test_accuracy = accuracy_score(X_test_prediction, Y_test)
X_test_accuracy

#Making a Prediction
#Input should contain [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age] features
input_data = (13,145,82,19,110,22.2,0.245,57)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
#standardize the input data
std_data = scaler.transform(input_data_reshaped)

prediction = classifier.predict(std_data)
print(prediction)