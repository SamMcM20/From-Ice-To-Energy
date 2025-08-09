"""
This script uses Random Forests with mass balance and precipitation to estmate energy production
A 1-year lag time is implemented into the code
The script has been adapted to work with a small dataset to reduce risk of over-fitting
"""

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import statsmodels.api as sm

# Configuration for Small Datasets 
# Reserve a fixed number of recent years for the final hold-out set
HOLD_OUT_YEARS = 3
# Set a minimum number of years to use for the initial training in CV
MIN_TRAIN_YEARS = 5


def create_lagged_features(df):
    """Creates lagged features for the given dataframe."""
    df_lagged = df.copy()
    df_lagged['massBalance_lag1'] = df_lagged['massBalance'].shift(1)
    df_lagged['precipitation_lag1'] = df_lagged['precipitation'].shift(1)
    return df_lagged



"""
STEP 1: LOAD DATA AND CREATE DEVELOPMENT / HOLD-OUT SETS
"""

print("STEP 1: Loading and Splitting Data for Small Dataset")

# Load Historical Data
try:
    full_df = pd.read_excel("INSERT LOCATION OF HISTORICAL DATA FILE")
    if len(full_df) >= 15:
        print(f"Warning: Dataset has {len(full_df)} years. This script is optimized for < 15 years.")
except FileNotFoundError:
    print("Historical data file not found.")

# Create lagged features on the entire dataset
full_df_lagged = create_lagged_features(full_df)

# Split into development and hold-out sets with a fixed number of years
development_df = full_df_lagged.iloc[:-HOLD_OUT_YEARS]
hold_out_df = full_df_lagged.iloc[-HOLD_OUT_YEARS:]

print(f"Full dataset size: {len(full_df_lagged)}")
print(f"Development set size: {len(development_df)}")
print(f"Hold-out set size: {len(hold_out_df)}")
print("-" * 40)



"""
STEP 2: TIME-SERIES "LEAVE-ONE-OUT" STYLE CROSS-VALIDATION
"""

print("STEP 2: Time-Series 'Leave-One-Out' Style Cross-Validation")
# Define features and prepare development data
features = ['massBalance', 'precipitation', 'massBalance_lag1', 'precipitation_lag1']
# .dropna() removes the first row with NaN from the initial lag
X_dev = development_df[features].dropna()
y_dev = development_df['energyProduction'].loc[X_dev.index]

# Model Simplification to Prevent Overfitting 
# With little data, we must constrain the model.
simplified_rf_params = {
    'n_estimators': 30,     # Fewer trees
    'max_depth': 4,         # Shallower trees
    'min_samples_leaf': 2,  # Must have at least 2 samples to form a leaf
    'random_state': 42
}
rf_model = RandomForestRegressor(**simplified_rf_params)

# Configure TimeSeriesSplit to test on one year at a time
n_splits = len(X_dev) - MIN_TRAIN_YEARS
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1)

# Manually loop through splits to calculate metrics
r2_scores, mae_scores, rmse_scores = [], [], []
for train_index, val_index in tscv.split(X_dev):
    X_train, X_val = X_dev.iloc[train_index], X_dev.iloc[val_index]
    y_train, y_val = y_dev.iloc[train_index], y_dev.iloc[val_index]

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_val)

    r2_scores.append(r2_score(y_val, y_pred))
    mae_scores.append(mean_absolute_error(y_val, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

print("Robust Performance Estimate from Cross-Validation:")
print(f"  Average R²: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores):.4f})")
print(f"  Average MAE: {np.mean(mae_scores):.2f}")
print(f"  Average RMSE: {np.mean(rmse_scores):.2f}")
print("-" * 40)



"""
STEP 3: FINAL MODEL TRAINING & FEATURE IMPORTANCE
"""

print("STEP 3: Final Model Training & Feature Importance")
# Train a final model on the ENTIRE development set
final_model = RandomForestRegressor(**simplified_rf_params)
final_model.fit(X_dev, y_dev)
print("Final model has been trained on the full development set.")

# Get and plot feature importances
importances = final_model.feature_importances_
feature_names = X_dev.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='teal')
plt.xlabel("Feature Importance")
plt.title("Final Model Feature Importance (Trained on Small Dataset)")
plt.gca().invert_yaxis()
plt.show()



"""
STEP 4: FINAL EVALUATION ON THE UNSEEN HOLD-OUT SET
"""

print("STEP 4: Final Evaluation on Hold-Out Set")
# Prepare the hold-out set
X_hold_out = hold_out_df[features].dropna()
y_hold_out = hold_out_df['energyProduction'].loc[X_hold_out.index]

if not X_hold_out.empty:
    y_pred_hold_out = final_model.predict(X_hold_out)
    hold_out_r2 = r2_score(y_hold_out, y_pred_hold_out)
    hold_out_mae = mean_absolute_error(y_hold_out, y_pred_hold_out)
    hold_out_rmse = np.sqrt(mean_squared_error(y_hold_out, y_pred_hold_out))
    print("Performance on UNSEEN Hold-Out Data:")
    print(f"  Hold-Out R²: {hold_out_r2:.4f}")
    print(f"  Hold-Out MAE: {hold_out_mae:.2f}")
    print(f"  Hold-Out RMSE: {hold_out_rmse:.2f}")
else:
    print("Hold-out set is empty after handling NaNs. Skipping final evaluation.")
print("-" * 40)

# Statistical significance test
X_dev_const = sm.add_constant(X_dev)
# Fit the Ordinary Least Squares (OLS) model
model_sm = sm.OLS(y_dev, X_dev_const).fit()
# Print the detailed summary
print(model_sm.summary())




"""
STEP 5: FORECASTING
"""

print("STEP 5: Forecasting")

try:
    futureGlacierDf = pd.read_excel("INSERT FUTURE GLACIER FORECAST")
    
    # Import precipitation and merge dataframes
    futurePrecipDf = pd.read_excel("INSERT FUTURE PRECIPITATION SCENARIO")
    futureDf = pd.merge(futureGlacierDf, futurePrecipDf[['year', 'precipitation']], on='year')
    
    # Fill potential NaNs
    futureDf['massBalance'].ffill(inplace=True)
    futureDf['massBalance'].bfill(inplace=True)
    futureDf['precipitation'].ffill(inplace=True)
    futureDf['precipitation'].bfill(inplace=True) 

    # Create lagged features
    futureDf = create_lagged_features(futureDf)
    futureDf['massBalance_lag1'].fillna(value=full_df['massBalance'].iloc[-1], inplace=True)
    futureDf['precipitation_lag1'].fillna(value=full_df['precipitation'].iloc[-1], inplace=True)

    # Predict future energy production using the final validated model
    X_future = futureDf[features]
    futureDf['predictedEnergy'] = final_model.predict(X_future)

    # Plot the final forecast
    plt.figure(figsize=(12, 6))
    plt.plot(futureDf['year'], futureDf['predictedEnergy'], label='Forecasted Energy', color='darkred')
    z = np.polyfit(futureDf['year'], futureDf['predictedEnergy'], 2)
    p = np.poly1d(z)
    plt.plot(futureDf['year'], p(futureDf['year']), "b--", label='Trendline')
    plt.xlabel('Year')
    plt.ylabel('Energy Production (GWh)')
    plt.title('RF Hydropower Forecast using Lag Time and Precipitation')
    plt.grid(True)
    plt.legend()
    plt.show()

except FileNotFoundError:
    print("\nFuture forecast file not found. Skipping forecast.")
