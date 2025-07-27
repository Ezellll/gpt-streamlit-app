import streamlit as st
from langchain_community.llms import OpenAI
import os

st.title("My First AI App")

openai_api_key = st.sidebar.text_input("OpenAI API Key")

def generate_response(prompt):
    llm = OpenAI(openai_api_key=openai_api_key,)

    st.info(llm(prompt))

with st.form("my_form"):
    text = st.text_area("Enter your prompt here:","Write three tips for learning Python")

    submit_button = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter a valid OpenAI API key.", icon="⚠️")
    if submit_button and openai_api_key.startswith("sk-"):
        generate_response(text)
