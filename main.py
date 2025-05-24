# mal_sentiment_predictor.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import streamlit as st

def main():
    # Load your custom MAL tokenizer and model
    with open('mal_tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    model = load_model('anime_review_model.h5')  # Your trained model

    def preprocess_text(text):
        """Convert raw text to model-ready input"""
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=500)  # Match your training max_len
        return padded

    # Streamlit App
    st.title('Anime Review Sentiment Analyzer')
    st.write('Enter an anime review to classify its sentiment (Positive/Negative)')

    # User input
    review_text = st.text_area('Anime Review:', height=150)
    
    if st.button('Analyze'):
        if review_text.strip():
            # Process and predict
            processed_input = preprocess_text(review_text)
            prediction = model.predict(processed_input)
            confidence = float(prediction[0][0])
            sentiment = 'Positive' if confidence > 0.4 else 'Negative'
            
            # Display results
            st.subheader('Result:')
            st.metric(label="Sentiment", value=sentiment)
            
            # Confidence gauge
            st.progress(confidence if sentiment == 'Positive' else 1-confidence)
            st.caption(f'Confidence: {confidence:.2%}')
            
            # Debug info (optional)
            with st.expander("Technical Details"):
                st.write(f"Raw prediction score: {confidence:.4f}")
                st.write(f"Model: SimpleRNN (trained on MAL reviews)")
        else:
            st.warning("Please enter a review first!")

if __name__ == "__main__":
    main()