import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def load_data():

    data = {
        'text': [
            'How do I reset my password?',
            'I cannot log into my account',
            'Help with login credentials',
            'forgot password',
            'reset login',
            'login help',
            'password reset',
            'cannot sign in',

            'How much does the subscription cost?',
            'What are your pricing plans?',
            'Is there a free trial?',
            'pricing plans',
            'how to pay',
            'pricing plan',
            'how much does it cost',
            'how much does it costs',
            'what is the price',
            'subscription price',

            'The app keeps crashing on launch',
            'I found a bug in the dashboard',
            'Error code 500 on checkout',
            'app is broken',
            'it is not working',
            'the app crashed',
            'bug in app',
            'crashing on start',

            'hello',
            'hi there',
            'hey',
            'good morning',
            'anybody home?',
            'hi'
        ],

        'category': [
            'Account/Auth',
            'Account/Auth',
            'Account/Auth',
            'Account/Auth',
            'Account/Auth',
            'Account/Auth',
            'Account/Auth',
            'Account/Auth',

            'Pricing/Billing',
            'Pricing/Billing',
            'Pricing/Billing',
            'Pricing/Billing',
            'Pricing/Billing',
            'Pricing/Billing',
            'Pricing/Billing',
            'Pricing/Billing',
            'Pricing/Billing',
            'Pricing/Billing',

            'Technical Bug',
            'Technical Bug',
            'Technical Bug',
            'Technical Bug',
            'Technical Bug',
            'Technical Bug',
            'Technical Bug',
            'Technical Bug',

            'Greeting',
            'Greeting',
            'Greeting',
            'Greeting',
            'Greeting',
            'Greeting'
        ]
    }

    return pd.DataFrame(data)

BOT_RESPONSES = {
    'Greeting': "Hello! I am Chat Bot AI. How can I help you today?",

    'Account/Auth':
    "To reset your password, click the 'Forgot Password' link on the login page or check your account settings.",

    'Pricing/Billing':
    "Our standard plan starts at $19/month. You can view the full pricing structure on our website.",

    'Technical Bug':
    "I'm sorry to hear that. Please send a screenshot of the error to our support team for investigation.",

    'Unknown':
    "I'm not quite sure I understand that. Could you please rephrase your request?"
}

def train_chatbot():

    df = load_data()

    X = df['text']
    y = df['category']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2)
        )),

        ('classifier', LogisticRegression(random_state=42))
    ])

    print("Training Chat Bot AI Model...")

    pipeline.fit(X, y)

    joblib.dump(pipeline, 'chatbot_model.pkl')

    print("Model saved successfully as 'chatbot_model.pkl'\n")

    return pipeline

def main():

    chatbot_model_path = 'chatbot_model.pkl'

    chatbot = train_chatbot()

    print("=" * 60)
    print("Chat Bot AI Initialized! (Type 'exit' to quit)")
    print("=" * 60)

    while True:

        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'exit':
            print("Bot: Goodbye! Have a great day.")
            break

        if not user_input:
            continue

        probabilities = chatbot.predict_proba([user_input])

        max_probability = max(probabilities[0])

        if max_probability < 0.25:
            intent = 'Unknown'
        else:
            intent = chatbot.predict([user_input])[0]

        response = BOT_RESPONSES[intent]

        print(f"Bot: {response}")

if __name__ == "__main__":
    main()