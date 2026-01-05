#To run the app use python -m streamlit run app.py
import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.utils import pad_sequences


#load trained model and tokenizer
@st.cache_resource #streamlit runs this fucntion only once when the server stats, without it
                   #the app would relead the AI model every time a button is pressed making it laggy.
def load_resources():
    model = tf.keras.models.load_model('sentiment_model.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer
model, tokenizer = load_resources()


st.title('Amazon Review Sentiment Analyser')
st.write('Enter an Amazon product reviews to see its sentiment.')
user_text = st.text_area('Review Text', height=150)

if st.button('Analyse Sentiment'):
    if user_text:
        sequences = tokenizer.texts_to_sequences([user_text])
        padded = pad_sequences(sequences, maxlen=100) #100 from preprocessing used during training model


        prediction = model.predict(padded)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.subheader('Prediction results:')

        if class_index == 0:
            st.error(f'Sentiment: Negative ({confidence:.2f}% confidence)')
        elif class_index == 1:
            st.warning(f'Sentiment: Neutral ({confidence:.2f}% confidence)')
        else:
            st.success(f'Sentiment: Positive ({confidence:.2f}% confidence)')

        
    else:
        st.error('Please enter some text')