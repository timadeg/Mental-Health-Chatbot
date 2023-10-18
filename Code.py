import pandas as pd

# Load the Mental Health in Tech Survey dataset
df = pd.read_csv('mental_health_survey.csv')

# Drop columns with missing values
df = df.dropna(axis=1)

# Remove outliers
df = df[df['Age'] >= 18]
df = df[df['Age'] <= 65]

import matplotlib.pyplot as plt
import seaborn as sns

# Correlation matrix heatmap
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm', annot=True)

# Distribution plots for key variables
sns.displot(df, x='Age')
sns.displot(df, x='Gender')
sns.displot(df, x='treatment')

from sklearn.preprocessing import OneHotEncoder

# Select relevant features
features = ['Age', 'Gender', 'treatment', 'mental_health_consequence', 'anonymity', 'coworkers']

# Encode categorical variables using OneHotEncoder
enc = OneHotEncoder()
enc.fit(df[features].astype(str))
encoded_features = enc.transform(df[features].astype(str)).toarray()

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Split dataset into training and testing sets
X_train, X_test = train_test_split(encoded_features, test_size=0.2)

# Train a collaborative filtering model using cosine similarity
similarity_matrix = cosine_similarity(X_train)

# Get recommendations for a user
user_id = 10
user_profile = X_train[user_id]
user_similarity = similarity_matrix[user_id]
user_ratings = X_train.T.dot(user_similarity)
recommended_items = list(df.iloc[user_ratings.argsort()[::-1]]['Item'])[:5]

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Define the chatbot interface
def start(update: Update, context: CallbackContext) -> None:
    context.bot.send_message(chat_id=update.effective_chat.id, text="Welcome to the Mental Health Recommender Chatbot! Please tell us about your mental health symptoms, triggers, and coping mechanisms.")

def message_handler(update: Update, context: CallbackContext) -> None:
    # Collect input data from the user
    user_input = update.message.text
    
    # Preprocess the input data
    encoded_input = enc.transform([user_input]).toarray()
    
    # Use the trained model to generate recommendations for the user
    user_similarity = cosine_similarity(encoded_input, X_train)[0]
    user_ratings = X_train.T.dot(user_similarity)
    recommended_items = list(df.iloc[user_ratings.argsort()[::-1]]['Item'])[:5]
    
    # Send recommendations to the user
    context.bot.send_message(chat_id=update.effective_chat.id, text=f"Based on your input, we recommend the following resources for you:\n{recommended_items}")
    
# Set up the chatbot handlers
updater = Updater(token='YOUR_TOKEN_HERE', use_context=True)
dispatcher = updater.dispatcher
dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, message_handler))

# Start the chatbot
updater.start_polling()

# Deploy the chatbot to a web platform or mobile app
# Collect feedback from users
