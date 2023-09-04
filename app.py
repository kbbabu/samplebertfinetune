import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import torch

@st.cache(allow_output_mutation = True)
def get_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('kbbabu/samplebertfinetune')
    return tokenizer, model

tokenizer, model = get_model()
user_input = st.text_area("Enter the text to Analyse")
button = st.button("Analyse")

d = {
    1:'Toxic',
    0:'Non Toxic'
}

if user_input and button:
    test_sample = tokenizer([user_input], padding = True, truncation = True, max_length = 512, return_tensors ='pt')
    output = model(**test_sample)
    st.write("Logits:",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis = 1)
    st.write("Prediction:", d[y_pred[0]])


