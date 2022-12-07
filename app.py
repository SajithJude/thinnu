# import asyncio
# import gc
import logging
import os
        # import os
import openai
# import pdfminer
# from pdfminer.high_level import extract_pages
from pdfminer.high_level import extract_text
import pandas as pd
import psutil 
import nltk
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords

import streamlit as st



os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def print_memory_usage():
    logging.info(f"RAM memory % used: {psutil.virtual_memory()[2]}")



def main():

    st.title("Upload a document and generate questions to stimulate a discussion from it..")
    num =   st.slider("how many questions do you want to generate ?",min_value=6,max_value=30)

    
    stop_words = set(stopwords.words('english'))

    tokeni = RegexpTokenizer('\w+')
    uploaded_file = st.file_uploader("Upload a file (less than 3000 words per doc)", "pdf")
    if uploaded_file is not None:
        tect = extract_text(uploaded_file)
        tokans = tokeni.tokenize(tect)
        filtered_sentence = [w for w in tokans if not w.lower() in stop_words]
  
        filtered_sentence = []
        
        for w in tokans:
            if w not in stop_words:
                filtered_sentence.append(w)
        my_lst_str = ' '.join(map(str, filtered_sentence))
        info = (my_lst_str[:3500] + '..') if len(my_lst_str) > 3500 else my_lst_str
        st.write(info)


    if st.button("Generate questions from document"):
        # print_memory_usage()

        openai.api_key =  os.getenv("OPENAI_API_KEY")


        response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Create a list of " + str(num) + " questions to stimulate discussion from the given text" + info  + "Include references to specific parts of the text in the questions, and invite students to refer to specific sentences from the text.",
        temperature=0.56,
        max_tokens=2066,
        top_p=1,
        frequency_penalty=0.35,
        presence_penalty=0,
        stop=["\n"]
        )
        st.write(response.choices[0].text)
        st.write(response)




if __name__ == "__main__":
    openai.api_key =  os.getenv("OPENAI_API_KEY")

    main()
