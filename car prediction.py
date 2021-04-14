import pandas as pd
import sklearn as sk

df = pd.read_csv('car data.csv')
print(df.head())
print(df.shape)
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())
print(df.isnull().sum())
print(df.describe())
print(df.columns)
final_dataset=df[[ 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
print(final_dataset.head())
final_dataset['current_year'] = 2020
final_dataset['no_years'] = final_dataset['current_year']-final_dataset['Year']
final_dataset.drop(['Year'], axis=1, inplace=True)
final_dataset.drop(['current_year'], axis=1, inplace=True)
final_dataset = pd.get_dummies(final_dataset,drop_first=True)
print(final_dataset.corr())
import seaborn as sns
sns.pairplot(final_dataset)
import matplotlib.pyplot as plt
corrmat = final_dataset.corr()
top_corr_feature = corrmat.index
plt.figure(figsize=(20, 20))
g=sns.heatmap(final_dataset[top_corr_feature].corr(),annot=True,cmap="RdYlGn")
#independent and dependent feature
x=final_dataset.iloc[:,1:]
y=final_dataset.iloc[:,0]
print(x.head())
print(y.head())
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
print(model.fit(x,y))
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train.shape)
from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()
import numpy as np
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(x_train,y_train)
print(rf_random.best_params_)

print(rf_random.best_score_)
predictions=rf_random.predict(x_test)
sns.displot(y_test-predictions)
plt.scatter(y_test,predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

import pickle
# open a file, where you ant to store the data
file = open('random_forest_regression_model.pkl', 'wb')

# dump information to that file
pickle.dump(rf_random, file)