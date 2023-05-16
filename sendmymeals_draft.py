import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load the data
data = pd.read_csv('/Users/amarahmed/Desktop/sendmymeals/sample_food_ingr.csv')

# Convert the order_datetime column to datetime
data['order_datetime'] = pd.to_datetime(data['order_datetime'])

# Extract features from the datetime
data['hour'] = data['order_datetime'].dt.hour
data['day'] = data['order_datetime'].dt.day
data['month'] = data['order_datetime'].dt.month
data['year'] = data['order_datetime'].dt.year

# Drop the 'order_datetime' column as it's no longer needed
data = data.drop(columns=['order_datetime'])

# Save the original 'dish' column before one-hot encoding
original_dish = data['dish'].copy()

# Perform one-hot encoding on the 'dish' column
data = pd.get_dummies(data, columns=['dish'])

# Add the original 'dish' column back to the DataFrame
data['dish'] = original_dish

# Get the list of unique ingredients
unique_ingredients = data['ingredient'].unique()

# Initialize an empty DataFrame to store the final predictions
predictions_df = pd.DataFrame()

# Loop through each unique ingredient
for ingredient in unique_ingredients:
    # Filter the data for the current ingredient
    ingredient_data = data[data['ingredient'] == ingredient]
    
    # Extract the features and target for the current ingredient
    features = ingredient_data.drop(columns=['ingredient', 'ingredient_amount', 'dish'])
    target = ingredient_data['ingredient_amount']

    # Fit a Random Forest model for the current ingredient
    model = RandomForestRegressor()
    model.fit(features, target)

    # Predict the amount of the current ingredient to order
    predictions = model.predict(features)
    
    # Compute the cost change
    old_cost = ingredient_data['ingredient_cost'].mean()
    new_cost = old_cost * (predictions / ingredient_data['ingredient_amount'].mean())
    cost_change = np.where(new_cost > old_cost, "up", "down")

    # Add the predictions to the final DataFrame
    ingredient_predictions = pd.DataFrame({
        'dish': ingredient_data['dish'],
        'ingredient': ingredient,
        'amount': predictions,
        'cost_change': cost_change
    })
    predictions_df = predictions_df.append(ingredient_predictions, ignore_index=True)

# Save the predictions DataFrame to a CSV file
predictions_df.to_csv('/Users/amarahmed/Desktop/sendmymeals/predictions.csv', index=False)
