import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE  # imblearn library can be installed using pip install imblearn
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Importing dataset and examining it
dataset = pd.read_csv("ChurnPrediction.csv")
pd.set_option('display.max_columns', None) # to make sure you can see all the columns in output window
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Converting Categorical features into Numerical features
dataset['PastEmployee'] = dataset['PastEmployee'].map({'Yes':1, 'No':0})
dataset['OverTime'] = dataset['OverTime'].map({'Yes':1, 'No':0})
dataset['Gender'] = dataset['Gender'].map({'Female':1, 'Male':0})
dataset['BusinessTravel'] = dataset['BusinessTravel'].map({'Non-Travel':0, 'Travel_Rarely':1, 'Travel_Frequently':2})
print(dataset.info())

categorical_features = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
final_data = pd.get_dummies(dataset, columns = categorical_features)
print(final_data.info())
print(final_data.head(2))

# Dividing dataset into label and feature sets
X = final_data.drop('PastEmployee', axis = 1) # Features
Y = final_data['PastEmployee'] # Labels
print(type(X))
print(type(Y))
print(X.shape)
print(Y.shape)

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X_scaled = feature_scaler.fit_transform(X)

# Implementing Multi-layer Perceptron (MLP)
# Tuning the MLP hyperparameters and implementing cross-validation using Grid Search
model = Pipeline([
        ('balancing', SMOTE(random_state = 101)),
        ('classification', MLPClassifier(hidden_layer_sizes = (50,), random_state=1)) # hidden_layer_sizes determines the number of layers and the number of units within those layers. It can be a hyperparameter
    ])
grid_param = {'classification__activation': ['logistic', 'tanh', 'relu'], 'classification__learning_rate_init': [.001,.01,.1,1,10,100], 'classification__max_iter' : [100,500,1000]}

gd_sr = GridSearchCV(estimator=model, param_grid=grid_param, scoring='recall', cv=5)

"""
In the above GridSearchCV(), scoring parameter should be set as follows:
scoring = 'accuracy' when you want to maximize prediction accuracy
scoring = 'recall' when you want to minimize false negatives
scoring = 'precision' when you want to minimize false positives
scoring = 'f1' when you want to balance false positives and false negatives (place equal emphasis on minimizing both)
"""

gd_sr.fit(X_scaled, Y)

best_parameters = gd_sr.best_params_
print(best_parameters)

best_result = gd_sr.best_score_ # Mean cross-validated score of the best_estimator
print(best_result)
