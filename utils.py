'''
import os
import google.generativeai as genai
from dotenv import load_dotenv
import joblib
import pickle


load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load local   ML model and vectorizer

def ml_predict(news_text):
    # Load your saved vectorizer and model
    vectorizer = joblib.load(r"A:\FAKE NEWS Detection\CODE\model.pkl")
    model = joblib.load(r"A:\FAKE NEWS Detection\CODE\vectorizer.pkl")

    # Vectorize the input text
    x = vectorizer.transform([news_text])  # : use vectorizer here
    # Make prediction
    prediction = model.predict(x)[0]
    return prediction

def gemini_judge(news_text):
    prompt = f"""
    You are a fake news detection expert. Analyze the following news article and respond only with "FAKE" or "REAL":
    
    \"\"\"{news_text}\"\"\"
    """
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    return response.text.strip().upper()
'''
#print(type(vectorizer))  
#print(type(model)) 

import os
import joblib
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load local ML model and vectorizer
vectorizer = joblib.load(r"A:\FAKE NEWS Detection\CODE\model.pkl")
model = joblib.load(r"A:\FAKE NEWS Detection\CODE\vectorizer.pkl")

# Global variable to store chat history
chat_history = []

def ml_predict(news_text):
    """
    Predict whether the news is FAKE or REAL using the ML model.
    """
    # Vectorize the input text
    x = vectorizer.transform([news_text])  # Use vectorizer to transform input
    # Make prediction using the model
    prediction = model.predict(x)[0]
    return "FAKE" if prediction == 1 else "REAL"  # Assuming 1 = FAKE, 0 = REAL

def gemini_judge(news_text):
    """
    Use the Gemini LLM to analyze the news and respond with FAKE or REAL.
    """
    global chat_history  # Use the global chat history

    # Add the user's input to the chat history
    chat_history.append(f"User: {news_text}")

    # Construct the prompt with the chat history
    prompt = "\n".join(chat_history) + f"\nBot: You are a fake news detection expert. Analyze the following news article and respond only with 'FAKE' or 'REAL':\n\n\"\"\"{news_text}\"\"\""

    # Generate the response using the Gemini LLM
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)

    # Add the bot's response to the chat history
    bot_response = response.text.strip().upper()
    chat_history.append(f"Bot: {bot_response}")

    # Return the bot's response
    return bot_response

def clear_chat_history():
    """
    Clear the chat history.
    """
    global chat_history
    chat_history = []