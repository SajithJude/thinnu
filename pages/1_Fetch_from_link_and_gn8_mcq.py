import streamlit as st


import openai
import os

st.title("Paste any URL and use it as a knowledge base to MCQ generate questions with answers..")
st.text("#adjust the slider  to fine tune the number of questions you want in the output")

url= st.text_input("enter the url you need to fetch data from")
num =   st.slider("how many questions do you want to generate ?",min_value=6,max_value=30)

if st.button("fetch data and generate mcq"):


    openai.api_key =  os.getenv("OPENAI_API_KEY")


    response = openai.Completion.create(
    model="text-davinci-002",
    prompt="Create a list of " + str(num) + " mcq questions with 4 answers from the given url " + url  + "specify the correct answer as well.",
    temperature=0.56,
    max_tokens=2066,
    top_p=1,
    frequency_penalty=0.35,
    presence_penalty=0,
    # stop=["\n"]
    )
    st.write(response.choices[0].text)
    st.write(response)
