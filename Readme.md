# AI ChatBot - Intent Classifier

A simple AI-powered chatbot built using Python and Machine Learning.

This project uses Natural Language Processing (NLP) techniques with TF-IDF Vectorization and Logistic Regression to classify user intents and return appropriate responses.

---

# Features

- Intent classification using Machine Learning
- Interactive command-line chatbot
- NLP text processing using TF-IDF
- Confidence threshold for unknown queries
- Multiple supported intents

---

# Supported Intents

- Greeting
- Account / Authentication Support
- Pricing / Billing Queries
- Technical Bug Reports

---

# Technologies Used

- Python
- pandas
- scikit-learn
- joblib

---

# Project Structure

```bash
AI-ChatBot/
│
├── app.py
├── requirements.txt
├── Readme.md
└── intent_model.pkl
```

---

# Installation & Setup

## 1. Clone Repository

```bash
git clone https://github.com/om-gunturkar/AI-ChatBot.git
```

---

## 2. Navigate to Project Folder

```bash
cd AI-ChatBot
```

---

## 3. Create Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate virtual environment:

```bash
.\venv\Scripts\Activate.ps1
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

```bash
python app.py
```

---

# Example Usage

```bash
You: hello
Bot: Hello! I am chatBotAI. How can I help you today?

You: forgot password
Bot: To reset your password, click the 'Forgot Password' link on the login page or check your account security settings.

You: pricing plans
Bot: Our standard plan starts at $19/month. You can view the full tier structure on our pricing page.
```

---

# How It Works

1. User enters text input
2. TF-IDF Vectorizer converts text into numerical features
3. Logistic Regression predicts the intent category
4. Chatbot returns a predefined response

---

# Future Improvements

- Add more intents and training data
- Build GUI using Flask or Streamlit
- Integrate OpenAI API
- Add database support
- Deploy chatbot online

---

# Author

Om Gunturkar
