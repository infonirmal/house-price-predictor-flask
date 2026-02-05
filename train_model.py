import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'area_sqft': [800, 900, 1000, 1200, 1500, 1800, 2000],
    'bedrooms': [2, 2, 2, 3, 4, 3, 5],
    'bathrooms': [1, 1, 2, 2, 3, 2, 4],
    'age': [20, 25, 30, 10, 5, 12, 3],
    'price': [140000, 160000, 200000, 300000, 400000, 320000, 600000]
}

df = pd.DataFrame(data)

X = df[['area_sqft', 'bedrooms', 'bathrooms', 'age']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

# Save model
with open('model/house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
