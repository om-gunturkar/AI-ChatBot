# chatBotAI: Intent-Based AI Chatbot

A lightweight, high-performance intent classification AI chatbot built using Python and `scikit-learn`. The bot parses user language inputs, classifies their underlying intent (Greetings, Authentication, Pricing, Technical Support), and maps them to dynamic responses.

## Technical Stack

- **Language:** Python 3.11
- **ML Framework:** Scikit-Learn
- **Data Engineering:** Pandas
- **Serialization:** Joblib

## Architectural Workflow

User Input -> TF-IDF Vectorizer -> Logistic Regression Classifier -> Confidence Threshold Verification -> Contextual Static Response Mapping

## How to Run Locally

1. Clone the repository and navigate into it:
   ```bash
   cd chatBotAI
   ```
