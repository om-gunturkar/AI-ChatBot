import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# ==========================================
# 1. DATA PREPARATION (Expanded Dataset)
# ==========================================
def load_data():
    """
    Creates an expanded dataset matching common user intents.
    Added singular/plural variations to help the model generalized better.
    """
    data = {
        'text': [
            # Account & Authentication Intents
            'How do I reset my password?', 'I cannot log into my account', 
            'Help with login credentials', 'forgot password', 'reset login',
            'login help', 'password reset', 'cannot sign in',
            
            # Pricing & Billing Intents
            'How much does the subscription cost?', 'What are your pricing plans?', 
            'Is there a free trial?', 'pricing plans', 'how to pay',
            'pricing plan', 'how much does it cost', 'how much does it costs',
            'what is the price', 'subscription price',
            
            # Technical Support Intents
            'The app keeps crashing on launch', 'I found a bug in the dashboard', 
            'Error code 500 on checkout', 'app is broken', 'it is not working',
            'the app crashed', 'bug in app', 'crashing on start',
            
            # Greetings & General Intents
            'hello', 'hi there', 'hey', 'good morning', 'anybody home?', 'hi'
        ],
        'category': [
            'Account/Auth', 'Account/Auth', 'Account/Auth', 'Account/Auth', 'Account/Auth', 'Account/Auth', 'Account/Auth', 'Account/Auth',
            'Pricing/Billing', 'Pricing/Billing', 'Pricing/Billing', 'Pricing/Billing', 'Pricing/Billing', 'Pricing/Billing', 'Pricing/Billing', 'Pricing/Billing', 'Pricing/Billing', 'Pricing/Billing',
            'Technical Bug', 'Technical Bug', 'Technical Bug', 'Technical Bug', 'Technical Bug', 'Technical Bug', 'Technical Bug', 'Technical Bug',
            'Greeting', 'Greeting', 'Greeting', 'Greeting', 'Greeting', 'Greeting'
        ]
    }
    return pd.DataFrame(data)

# ==========================================
# 2. CHATBOT STATIC RESPONSES
# ==========================================
BOT_RESPONSES = {
    'Greeting': "Hello! I am chatBotAI. How can I help you today?",
    'Account/Auth': "To reset your password, click the 'Forgot Password' link on the login page or check your account security settings.",
    'Pricing/Billing': "Our standard plan starts at $19/month. You can view the full tier structure on our pricing page.",
    'Technical Bug': "I'm sorry to hear that. Please forward a screenshot of the error to our technical support desk so we can investigate.",
    'Unknown': "I'm not quite sure I understand that. Could you please rephrase your request?"
}

# ==========================================
# 3. AI MODEL TRAINING PIPELINE
# ==========================================
def train_model():
    df = load_data()
    X = df['text']
    y = df['category']
    
    # We keep ngram_range=(1,2) to capture both individual words and two-word phrases
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    print("Training the AI Intent Model...")
    pipeline.fit(X, y)
    
    joblib.dump(pipeline, 'intent_model.pkl')
    print("Model saved successfully as 'intent_model.pkl'\n")
    return pipeline

# ==========================================
# 4. INTERACTIVE CHAT INTERFACE
# ==========================================
def main():
    model_path = 'intent_model.pkl'
    
    # Force retrain to pick up the new updated data strings
    pipeline = train_model()
    
    print("=" * 60)
    print("chatBotAI Initialized! (Type 'exit' to quit)")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("Bot: Goodbye! Have a great day.")
            break
            
        if not user_input:
            continue
            
        probabilities = pipeline.predict_proba([user_input])
        max_prob = max(probabilities[0])
        
        # Optimized Threshold: Reduced to 0.25 to make the bot more flexible 
        # with small variations while still catching total gibberish.
        if max_prob < 0.25:
            intent = 'Unknown'
        else:
            intent = pipeline.predict([user_input])[0]
        
        response = BOT_RESPONSES[intent]
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()