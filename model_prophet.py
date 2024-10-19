from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import cross_validation, performance_metrics

from sklearn.metrics import mean_absolute_error

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools

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

dataset = dataset[(dataset.store == 3) & (dataset.item == 4)][["date", 'sales', 'weekend', 'weekday']]

dataset = dataset.rename(columns={'date': 'ds',
                        'sales': 'y', 
                        'holidays': 'hols'})

dataset['ds'] = pd.to_datetime(dataset['ds'])
dataset.head()

train_set = dataset[(dataset.ds<='2017-01-01')]
test_set = dataset[(dataset.ds>'2017-01-01')]

param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'seasonality_mode':['additive', 'multiplicative']
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params)
    m.add_regressor('weekend')
    m.add_country_holidays(country_name='IN')
    m.add_regressor('weekday')
    m.fit(train_set)  # Fit model with given params
    df_cv = cross_validation(m, initial='730 days', period='365 days', horizon='730 days')
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)

best_params = all_params[np.argmin(rmses)]
print(best_params)

m = Prophet(**best_params)
m.add_regressor('weekend')
m.add_country_holidays(country_name='IN')
m.add_regressor('weekday')
m.fit(train_set)

# import pickle
# # Save the model to a file
# with open('prophet_model.pkl', 'wb') as f:
    # pickle.dump(m, f)

forecast = m.predict(test_set.drop('y', axis=1))
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

Y_pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
y_test = test_set[['ds', 'y']]

merged_test_set = pd.merge(y_test, Y_pred, on='ds')
test_mae = mean_absolute_error(merged_test_set['y'].values, merged_test_set['yhat'].values)

print(f"Test MAE: {test_mae}")

# bland altman plot

# Calculate the mean of the two methods (measurements and predictions)
mean_values = np.mean([merged_test_set['y'].values, merged_test_set['yhat'].values], axis=0)

# Calculate the difference between the measurements and predictions
difference = merged_test_set['y'].values - merged_test_set['yhat'].values

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
plt.scatter(merged_test_set['ds'], merged_test_set['yhat'].values, color='blue', label='prediction')
plt.scatter(merged_test_set['ds'], merged_test_set['y'].values, color='green', label='observation')
plt.xticks([merged_test_set['ds'].min(), merged_test_set['ds'].max()])
plt.title('Sales Trend Plot')
plt.xlabel('Date')
plt.ylabel('Sales (Predicted and Observed)')
plt.legend()
plt.grid(axis='y', which='both', alpha=0.2)
plt.show()