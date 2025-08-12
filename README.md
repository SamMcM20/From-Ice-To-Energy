The Linear Regression (LR) and Random Forest (RF) models are intended to be used with the provided data from the GLomfjord and Svartisen power stations in Norway's Nordland region, in combination with the data from the Engabreen glacier and regional precipitation.
The models are written to use 'massBalance' as the glacier change variable, but summer mass balance and length change files have been included. Minor updates to the variable names within the model will allow for smooth operation. 
Alternatively, the headings above the alternative variable values in the corresponding Excel sheets can be changed to 'massBalance', allowing the code to run with no errors.

The purpose of these models is to identify correlations between energy production and historical glacier changes. Future mass balance and precipitation files are included for future modeling of energy production.
Future precipitation data is stored in the 'CMIP' folder, and comes in the form of the SSP2 and SSP5 scenarios, with SSP2 being a middle-of-the-road prediction and SSP5 being a worst-case climate crisis prediction.
Model variables are: MB=Mass Balance, PR=Precipitation, LAG=Lagged Variables. 
It is worth noting that testing shows the GLomfjord data run through the 'LR_MB' and 'RF_MB_PR_LAG' models produce the strongest results.

It is recommended to download all of the included files to a single folder, then choose the desired model and update the historical data, future glacier, and future precipitation file paths where appropriate. These lines are marked accordingly in 'Stage 1' and 'Stage 5' of the script.
