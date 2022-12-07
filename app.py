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
# from PIL import Image
# from streamlit import components
# import transformers_interpret
# import transformers
# from streamlit.caching import clear_cache
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from transformers_interpret import SequenceClassificationExplainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def print_memory_usage():
    logging.info(f"RAM memory % used: {psutil.virtual_memory()[2]}")


# @st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
# def load_model(model_name):
#     return (
#         AutoModelForSequenceClassification.from_pretrained(model_name),
#         AutoTokenizer.from_pretrained(model_name),
#     )


def main():

    st.title("Upload a CV and check if the canditate is relevant to the role..")

    
    stop_words = set(stopwords.words('english'))

    tokeni = RegexpTokenizer('\w+')
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    if uploaded_file is not None:
        tect = extract_text(uploaded_file)
        tokans = tokeni.tokenize(tect)
        filtered_sentence = [w for w in tokans if not w.lower() in stop_words]
  
        filtered_sentence = []
        
        for w in tokans:
            if w not in stop_words:
                filtered_sentence.append(w)
        my_lst_str = ' '.join(map(str, filtered_sentence))
        info = (my_lst_str[:350] + '..') if len(my_lst_str) > 350 else my_lst_str
        st.write(info)


    if st.button("Interpret document"):
        # print_memory_usage()


        openai.api_key = "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"

        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=info,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
        )
        st.write(response)
        # st.text("Output")
        # with st.spinner("Interpreting your text (This may take some time)"):
        #     if explanation_class_choice != "predicted":
        #         word_attributions = cls_explainer(
        #             info,
        #             class_name=explanation_class_choice,
        #             embedding_type=emb_type_num,
        #             internal_batch_size=2,
        #         )
        #     else:
        #         word_attributions = cls_explainer(
        #             info, embedding_type=emb_type_num, internal_batch_size=2
        #         )

        # if word_attributions:
        #     word_attributions_expander = st.beta_expander(
        #         "Click here for raw word attributions"
        #     )
        #     with word_attributions_expander:
        #         st.json(word_attributions)
        #     components.v1.html(
        #         cls_explainer.visualize()._repr_html_(), scrolling=True, height=350
        #     )


if __name__ == "__main__":
    main()
