import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
# model=load_model('next_word_lstm.h5')
model=load_model('next_word_lstm.keras')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# SOLUTION 3: Improved Prediction Function with Temperature Sampling

def predict_next_word_improved(model, tokenizer, text, max_sequence_len, temperature=1.0, top_k=5):
    """
    Improved prediction function with temperature sampling and top-k filtering
    
    Args:
        temperature: Controls randomness (0.5=conservative, 1.0=normal, 1.5=creative)
        top_k: Number of top candidates to consider
    """
    # Clean and tokenize input text
    text = text.strip().lower()
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    if len(token_list) == 0:
        return "No valid words found"
    
    # Prepare sequence
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Get prediction probabilities
    predicted = model.predict(token_list, verbose=0)[0]
    
    # Apply temperature scaling
    if temperature != 1.0:
        predicted = np.log(predicted + 1e-8) / temperature
        predicted = np.exp(predicted) / np.sum(np.exp(predicted))
    
    # Get top-k predictions
    top_indices = np.argsort(predicted)[-top_k:][::-1]
    top_probs = predicted[top_indices]
    
    # Normalize probabilities for top-k
    top_probs = top_probs / np.sum(top_probs)
    
    # Sample from top-k candidates
    chosen_idx = np.random.choice(top_indices, p=top_probs)
    
    # Find the word
    for word, index in tokenizer.word_index.items():
        if index == chosen_idx:
            return word, predicted[chosen_idx], top_indices, predicted[top_indices]
    
    return "Unknown", 0.0, [], []



# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    next_word_improved , prob, _, _ = predict_next_word_improved(model, tokenizer, input_text, max_sequence_len, temperature=5)
    st.write(f'Next word: {next_word}')
    st.write(f'With Temperature Sampling: {next_word_improved}')
    st.write(f'Probability: {prob}')

