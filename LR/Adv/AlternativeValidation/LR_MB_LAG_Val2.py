"""
This script uses Linear Regression with mass balance to estimate energy output.
The script implements a 1-year lag time dataframe.
It has been tailored to work with a small datset by validating on the full dataset.
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm

# Configuration for Small Datasets
# Set a minimum number of years to use for the initial training
MIN_TRAIN_YEARS = 5

# Creates lagged mass balance
def create_lagged_features(df):
    df_lagged = df.copy()
    df_lagged['massBalance_lag1'] = df_lagged['massBalance'].shift(1)
    return df_lagged


"""
STEP 1: LOAD AND PREPARE DATA
"""

print("---- STEP 1: Loading and Preparing Data ----")

# Load Historical Data
try:
    full_df = pd.read_excel("INSERT LOCATION OF HISTORICAL DATA FILE")

    if len(full_df) >= 15:
        print(f"Warning: Dataset has {len(full_df)} years. This script is optimized for < 15 years.")

except FileNotFoundError:
    print("Historical data file not found.")

# Create lagged features on the entire dataset
full_df_lagged = create_lagged_features(full_df)

print(f"Full dataset size: {len(full_df_lagged)}")
print("-" * 40)


"""
STEP 2: TIME-SERIES "LEAVE-ONE-OUT" STYLE CROSS-VALIDATION
"""

print("---- STEP 2: Time-Series Cross-Validation on Full Dataset ----")
# Define features and prepare the full historical dataset for validation
features = ['massBalance', 'massBalance_lag1']
X_hist = full_df_lagged[features].dropna()
y_hist = full_df_lagged['energyProduction'].loc[X_hist.index]

# Initialize Linear Regression Model
lr_model = LinearRegression()

# Configure TimeSeriesSplit to test on one year at a time
n_splits = len(X_hist) - MIN_TRAIN_YEARS
if n_splits < 1:
    raise ValueError("Not enough data for cross-validation after lagging.")
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1)

# Manually loop through splits to calculate metrics
r2_scores, mae_scores, rmse_scores = [], [], []
for train_index, val_index in tscv.split(X_hist):
    X_train, X_val = X_hist.iloc[train_index], X_hist.iloc[val_index]
    y_train, y_val = y_hist.iloc[train_index], y_hist.iloc[val_index]

    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_val)

    r2_scores.append(r2_score(y_val, y_pred))
    mae_scores.append(mean_absolute_error(y_val, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

print("Robust Performance Estimate from Cross-Validation:")
print(f"  Average R²: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
print(f"  Average MAE: {np.mean(mae_scores):.2f}")
print(f"  Average RMSE: {np.mean(rmse_scores):.2f}")
print("-" * 40)


"""
STEP 3: FINAL MODEL TRAINING & COEFFICIENT ANALYSIS
"""

print("---- STEP 3: Final Model Training & Coefficient Analysis ----")
# Train a final model on the ENTIRE historical dataset
final_model = LinearRegression()
final_model.fit(X_hist, y_hist)
print("Final model has been trained on the full historical dataset.")

# Get and plot the model's coefficients as a measure of feature importance
coefficients = final_model.coef_
feature_names = X_hist.columns
plt.figure(figsize=(10, 6))
# Use the absolute value of the coefficients for ranking importance
plt.barh(feature_names, np.abs(coefficients), color='royalblue')
plt.xlabel("Absolute Coefficient Value (Feature Importance)")
plt.title("Linear Regression Model Coefficients")
plt.gca().invert_yaxis()
plt.show()

"""
STEP 4: STATISTICAL SIGNIFICANCE TEST
"""
print("---- STEP 4: Statistical Significance Test ----")
# Statistical significance test on the full historical dataset
X_hist_const = sm.add_constant(X_hist)
# Fit the Ordinary Least Squares (OLS) model
model_sm = sm.OLS(y_hist, X_hist_const).fit()
# Print the detailed summary
print(model_sm.summary())


"""
STEP 5: FORECASTING
"""
print("---- STEP 5: Forecasting ----")

try:
    futureDf = pd.read_excel("INSERT FUTURE GLACIER FORECAST")
    
    # Proactively handle NaNs that may exist in the source Excel file
    futureDf['massBalance'].ffill(inplace=True)
    futureDf['massBalance'].bfill(inplace=True)

    # Create the lagged feature, which introduces a single NaN in the first row.
    futureDf = create_lagged_features(futureDf)
    
    # Fill the NaN created by the lag using the last clean historical value.
    last_historical_mass_balance = X_hist['massBalance'].iloc[-1]
    futureDf['massBalance_lag1'].fillna(value=last_historical_mass_balance, inplace=True)

    # Predict future energy production using the final validated model
    X_future = futureDf[features]
    futureDf['predictedEnergy'] = final_model.predict(X_future)
    

    # Plot the final forecast
    plt.figure(figsize=(12, 6))
    plt.plot(futureDf['year'], futureDf['predictedEnergy'], label='Forecasted Energy', color='darkred')
    z = np.polyfit(futureDf['year'], futureDf['predictedEnergy'], 1) # A linear trendline for a linear model
    p = np.poly1d(z)
    plt.plot(futureDf['year'], p(futureDf['year']), "b--", label='Trendline')
    plt.xlabel('Year')
    plt.ylabel('Energy Production (GWh)')
    plt.title('LR Hydropower Forecast using Lag Time and no Precipitation')
    plt.grid(True)
    plt.legend()
    plt.show()

except FileNotFoundError:
    print("\nFuture forecast file not found. Skipping forecast.")