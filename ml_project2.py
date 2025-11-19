import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("Multi.csv")
X = df[["area", "bedrooms", "age"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

new_house = pd.DataFrame([[2500, 4, 2]], columns=["area", "bedrooms", "age"])
predicted_price = model.predict(new_house)
print(f"\nPredicted price for 2500 sqft, 4 bedrooms, 2 years old: ₹{predicted_price[0]:,.2f}")

try:
    a = int(input("\nEnter area (sqft): "))
    b = int(input("Enter number of bedrooms: "))
    c = int(input("Enter age of house (years): "))
except ValueError:
    print("Invalid input. Please enter numeric values.")
    exit()

user_input = pd.DataFrame([[a, b, c]], columns=["area", "bedrooms", "age"])
user_price = model.predict(user_input)[0]

print(f"Your predicted house price is: ₹{user_price:,.2f}")


user_input = pd.DataFrame([[a, b, c]], columns=["area", "bedrooms", "age"])
user_price = model.predict(user_input)[0]

print(f"Your predicted house price is: ₹{user_price:,.2f}")

import pickle
pickle.dump(model,open("Houseprediction.pkl","wb"))