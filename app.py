import streamlit as st
import torch
import sys
import os
# add src path
project_root = os.path.abspath(".")
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)
from model_v2 import SentimentModel
# ---- CONFIG ----
MAX_LEN = 200
VOCAB_PATH = "word2idx.pt"
MODEL_PATH = "sentiment_model.pth"
# ---- LOAD VOCAB ----
word2idx = torch.load(VOCAB_PATH)
# ---- LOAD MODEL ----
vocab_size = len(word2idx)
embed_dim = 128
hidden_dim = 128
model = SentimentModel(vocab_size, embed_dim, hidden_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
# ---- CLEAN FUNCTION ----
import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
# ---- PREDICT FUNCTION ----
def predict(text):
    text = clean_text(text)
    tokens = text.split()
    encoded = [
        word2idx.get(token, word2idx['<UNK>'])
        for token in tokens
    ]
    if len(encoded) < MAX_LEN:
        encoded += [0] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]
    input_tensor = torch.tensor(encoded).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = (output >= 0).float().item()
    return "Positive" if pred == 1 else "Negative"
# ---- UI ----
st.title("🎬 Sentiment Analysis App")
user_input = st.text_area("Enter a movie review:")
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        result = predict(user_input)
        st.success(f"Prediction: {result}")