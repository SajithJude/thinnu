import streamlit as st


import openai
import os


url= st.text_input("enter the url you need to fetch data from")

if st.button("analyze and sumamarize"):


    openai.api_key =  os.getenv("OPENAI_API_KEY")


    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="fetch content from this link "+url +"and Summarize it for a second-grade student.",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    st.write(response.choices[0].text)
    # st.write(response)
