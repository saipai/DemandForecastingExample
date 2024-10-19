import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
random.seed(42)

import warnings 
warnings.filterwarnings('ignore')


filepath = "data/train.csv"
dataset = pd.read_csv(filepath)

# split the date into strings for year, month and day
parts = dataset["date"].str.split("-", n = 3, expand = True)
dataset["year"]= parts[0].astype('int')
dataset["month"]= parts[1].astype('int')
dataset["day"]= parts[2].astype('int')
print(dataset.head(3))

from datetime import date, datetime
import holidays

# what day of the week is it
def which_day(year, month, day): 
    
    d = datetime(year,month,day) 
    return d.weekday()

dataset['weekday'] = dataset.apply(lambda x: which_day(x['year'],
                                                   x['month'],
                                                   x['day']),
                               axis=1)
print(dataset.head(3))

# is today a weekday ot not
def weekend_or_weekday(year,month,day):
    d = datetime(year,month,day)
    if d.weekday()>4: # friday == 4
        return 1
    else:
        return 0

dataset['weekend'] = dataset.apply(lambda x:weekend_or_weekday(x['year'], x['month'], x['day']), axis=1)
print(dataset.head(3))

# is today a holiday
def is_holiday(x):
  
  india_holidays = holidays.country_holidays('IN')

  if india_holidays.get(x):
    return 1
  else:
    return 0

dataset['holidays'] = dataset['date'].apply(is_holiday)
print(dataset.head(3))

# create a new data frame
new_dataset = pd.DataFrame([])
for location in dataset.store.unique(): # for every store
    for product in dataset.item.unique(): # for every product
        # last 7 day sales
        t_7 = []
        t_6 = []
        t_5 = []
        t_4 = []
        t_3 = []
        t_2 = []
        t_1 = []
        temp = dataset[(dataset.store == location) & (dataset.item ==product)]
        temp.sort_values('date')

        for time_step in range(len(temp)):
            if time_step<7:
                    t_7.append(np.nan)
                    t_6.append(np.nan)
                    t_5.append(np.nan)
                    t_4.append(np.nan)
                    t_3.append(np.nan)
                    t_2.append(np.nan)
                    t_1.append(np.nan)
            else:
                    t_7.append(int(temp.iloc[time_step-7,:]['sales']))
                    t_6.append(int(temp.iloc[time_step-6,:]['sales']))
                    t_5.append(int(temp.iloc[time_step-5,:]['sales']))
                    t_4.append(int(temp.iloc[time_step-4,:]['sales']))
                    t_3.append(int(temp.iloc[time_step-3,:]['sales']))
                    t_2.append(int(temp.iloc[time_step-2,:]['sales']))
                    t_1.append(int(temp.iloc[time_step-1,:]['sales']))

        temp['t_1'] = t_1
        temp['t_2'] = t_2
        temp['t_3'] = t_3
        temp['t_4'] = t_4
        temp['t_5'] = t_5
        temp['t_6'] = t_6

        if len(new_dataset) == 0:
            new_dataset = temp
        else:
            new_dataset = pd.concat([new_dataset, temp], axis=0, ignore_index=True)

print(new_dataset.head(3))

# let us limit our analysis to an item from one store location
# this can be expanded within the same model for ML 
# or by building different models for each store and product as required
new_dataset = new_dataset[(new_dataset.store == 3) & (new_dataset.item == 4)]

# drop missing values
new_dataset.dropna(inplace=True)

# get test set
test_set = new_dataset[(new_dataset.date>'2017-01-01')]

# drop test set from the dataset
new_dataset = new_dataset[(new_dataset.date<='2017-01-01')]

# split into train and dev sets
X_train = new_dataset.drop(["date", "sales"], axis=1)
y_train = new_dataset["sales"]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Defining which features to scale
features_list = X_train.columns

# scale only trailing 7-day sales features
scaled_features = []
for feature in features_list:
    if "t_" in feature:
        scaled_features.append(feature)

non_scaled_features = list(set(features_list)-set(scaled_features))

preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), scaled_features),
        ('passthrough', 'passthrough', non_scaled_features)
    ]
)

# Pipeline with scaling of a few features and a random forest model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Hyperparameter grid for the random forest model and pipeline
param_grid = {
    'model__n_estimators': [5, 10, 20, 30, 50],  # Number of trees in the forest
    'model__max_depth': [None, 5, 10, 30],     # Maximum depth of the tree
    'model__max_features':[0.1, 0.3, 0.7, 1, 'sqrt', 'log2']
}

from sklearn.model_selection import GridSearchCV

# Set up GridSearchCV
mdl_pipe = GridSearchCV(pipeline, param_grid, cv=5,
                        scoring='neg_mean_squared_error', refit=True, verbose=3)

# Train the model with grid search
mdl_pipe.fit(X_train, y_train)



from sklearn.metrics import root_mean_squared_error, mean_absolute_error

# Evaluate the model on test data
y_pred = mdl_pipe.predict(test_set.drop(["date", "sales"], axis=1))
y_test = test_set["sales"].values
test_mse = root_mean_squared_error(y_test, y_pred)
test_mae = mean_absolute_error(y_test, y_pred)
print(f"Test MSE: {test_mse}")
print(f"Test MAE: {test_mae}")

# add predictions to the test set
test_set["predictions"] = y_pred

# bland altman plot

# Calculate the mean of the two methods (measurements and predictions)
mean_values = np.mean([y_test, y_pred], axis=0)

# Calculate the difference between the measurements and predictions
difference = y_test - y_pred

# Calculate the mean difference and limits of agreement
mean_diff = np.mean(difference)  # Mean difference
std_diff = np.std(difference)    # Standard deviation of the differences
loa_upper = mean_diff + 1.96 * std_diff  # Upper limit of agreement (Mean + 1.96 * SD)
loa_lower = mean_diff - 1.96 * std_diff  # Lower limit of agreement (Mean - 1.96 * SD)

# Create the Bland-Altman plot
plt.figure(figsize=(8, 6))
plt.scatter(mean_values, difference, alpha=0.75, color='blue')  # Mean vs Difference scatter plot
plt.axhline(mean_diff, color='red', linestyle='--', label=f'Mean Difference: {mean_diff:.2f}')  # Mean difference line
plt.axhline(loa_upper, color='green', linestyle='--', label=f'$\mu$+2$\sigma$: {loa_upper:.2f}')  # Upper limit of agreement
plt.axhline(loa_lower, color='green', linestyle='--', label=f'$\mu$-2$\sigma$: {loa_lower:.2f}')  # Lower limit of agreement
plt.title('Bland-Altman Plot')
plt.xlabel('Mean of Measurements and Predictions')
plt.ylabel('Difference Between Measurements and Predictions')
plt.legend()
plt.grid(True)


# Create the trend plot
plt.figure(figsize=(6, 4))
plt.scatter(test_set['date'], test_set['predictions'].values, color='blue', label='prediction')
plt.scatter(test_set['date'], test_set['sales'].values, color='green', label='observation')
plt.xticks([test_set['date'].min(), test_set['date'].max()])
plt.title('Sales Trend Plot')
plt.xlabel('Date')
plt.ylabel('Sales (Predicted and Observed)')
plt.legend()
plt.grid(axis='y', which='both', alpha=0.2)
plt.show()