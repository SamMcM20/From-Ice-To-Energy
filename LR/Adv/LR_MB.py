"""
This script uses Linear Regression with mass balance and precipitation to estimate energy output.
It has been tailored to work with a small datset to reduce over fitting.
"""


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression # Changed import
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm

# Configuration for Small Datasets
# Reserve a fixed number of recent years for the final hold-out set
HOLD_OUT_YEARS = 3
# Set a minimum number of years to use for the initial training in CV
MIN_TRAIN_YEARS = 5


"""
STEP 1: LOAD DATA AND CREATE DEVELOPMENT / HOLD-OUT SETS
"""

print("---- STEP 1: Loading and Splitting Data for Small Dataset ----")

# Load Historical Data
try:
    full_df = pd.read_excel("INSERT LOCATION OF HISTORICAL DATA FILE")
    if len(full_df) >= 15:
        print(f"Warning: Dataset has {len(full_df)} years. This script is optimized for < 15 years.")
        
    full_df.dropna(subset=['massBalance', 'energyProduction'], inplace=True)
        
except FileNotFoundError:
    print("Historical data file not found.")

# Split into development and hold-out sets with a fixed number of years
development_df = full_df.iloc[:-HOLD_OUT_YEARS]
hold_out_df = full_df.iloc[-HOLD_OUT_YEARS:]

print(f"Full dataset size: {len(full_df)}")
print(f"Development set size: {len(development_df)}")
print(f"Hold-out set size: {len(hold_out_df)}")
print("-" * 40)



"""
STEP 2: TIME-SERIES "LEAVE-ONE-OUT" STYLE CROSS-VALIDATION
"""

print("---- STEP 2: Time-Series 'Leave-One-Out' Style Cross-Validation ----")
# Define features and prepare development data (only massBalance)
features = ['massBalance']
X_dev = development_df[features]
y_dev = development_df['energyProduction']

# Initialize Linear Regression Model 
lr_model = LinearRegression() 

# Configure TimeSeriesSplit to test on one year at a time
n_splits = len(X_dev) - MIN_TRAIN_YEARS
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=1)

# Manually loop through splits to calculate metrics
r2_scores, mae_scores, rmse_scores = [], [], []
for train_index, val_index in tscv.split(X_dev):
    X_train, X_val = X_dev.iloc[train_index], X_dev.iloc[val_index]
    y_train, y_val = y_dev.iloc[train_index], y_dev.iloc[val_index]

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
STEP 3: FINAL MODEL TRAINING
"""

print("---- STEP 3: Final Model Training ----")
# Train a final model on the ENTIRE development set
final_model = LinearRegression() 
final_model.fit(X_dev, y_dev)
print("Final model has been trained on the full development set.")



"""
STEP 4: FINAL EVALUATION ON THE UNSEEN HOLD-OUT SET
"""

print("---- STEP 4: Final Evaluation on Hold-Out Set ----")
# Prepare the hold-out set
X_hold_out = hold_out_df[features]
y_hold_out = hold_out_df['energyProduction']

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
    print("Hold-out set is empty. Skipping final evaluation.")
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
print("---- STEP 5: Forecasting ----")
try:
    futureDf = pd.read_excel("INSERT FUTURE GLACIER FORECAST")
    
    # Proactively handle potential NaNs in the source Excel file
    futureDf['massBalance'].ffill(inplace=True)
    futureDf['massBalance'].bfill(inplace=True)

    # Predict future energy production using the final validated model
    X_future = futureDf[features]
    futureDf['predictedEnergy'] = final_model.predict(X_future)

    # Plot the final forecast
    plt.figure(figsize=(12, 6))
    plt.plot(futureDf['year'], futureDf['predictedEnergy'], label='Forecasted Energy', color='darkred')
    z = np.polyfit(futureDf['year'], futureDf['predictedEnergy'], 1) # Linear trendline for a linear model
    p = np.poly1d(z)
    plt.plot(futureDf['year'], p(futureDf['year']), "b--", label='Trendline')
    plt.xlabel('Year')
    plt.ylabel('Energy Production (GWh)')
    plt.title('LR Hydropower Forecast using no Precipitation or Lag Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    output_columns = ['year', 'predictedEnergy']
    output_df = futureDf[output_columns]
    
    # Define the output file path with a filename and .xlsx extension
    output_filepath = "HydropowerForecastResults.xlsx"
    
    # Export the DataFrame to an Excel file
    output_df.to_excel(output_filepath, index=False)
    
    print(f"\nForecast results successfully exported to: {output_filepath}")

except FileNotFoundError:

    print("\nFuture forecast file not found. Skipping forecast.")
