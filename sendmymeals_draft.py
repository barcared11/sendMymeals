import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import openai
from openai.api_resources.completion import Completion

# Load the data
data = pd.read_csv('/Users/amarahmed/Desktop/sample_food_ingr.csv')

# Convert the order_datetime column to datetime
data['order_datetime'] = pd.to_datetime(data['order_datetime'])

# Extract features from the datetime
data['hour'] = data['order_datetime'].dt.hour
data['day'] = data['order_datetime'].dt.day
data['month'] = data['order_datetime'].dt.month
data['year'] = data['order_datetime'].dt.year

# Extract the features and target
features = data[['dish', 'ingredient', 'cost', 'dish_sold', 'hour', 'day', 'month', 'year', 'wasted']]
target = data['ingredient_amount']

# Fit a Random Forest model
model = RandomForestRegressor()
model.fit(features, target)

# Predict the amount of each ingredient to order
predictions = model.predict(features)

# Write the predictions to a new csv file
predictions_df = pd.DataFrame(predictions, columns=['ingredient_amount'])
predictions_df.to_csv('predictions.csv', index=False)

# Connect to the ChatGPT API
openai.api_key = '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'

# Prepare a message for the ChatGPT API
message = {
    'role': 'system',
    'content': 'You are a helpful assistant.'
}

# Send the message to the ChatGPT API
response = Completion.create(engine="text-davinci-002", model="gpt-4.0-turbo", messages=[message], max_tokens=60)

# Get the response from the ChatGPT API
response_text = response['choices'][0]['message']['content']

# Print the response
print(response_text)