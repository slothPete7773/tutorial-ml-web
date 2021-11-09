import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv("./data/FuelConsumption.csv")

# Recreate features
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Extract data from df
x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]

# Model Instantiation
regressor = LinearRegression()

# Fitting the model with extracted data
regressor.fit(x, y)

pickle.dump(regressor, open('model.pkl', 'wb'))
