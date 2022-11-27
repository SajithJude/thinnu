import asyncio
import gc
import logging
import os
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
from PIL import Image
from streamlit import components
# import transformers_interpret
# import transformers
# from streamlit.caching import clear_cache
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def print_memory_usage():
    logging.info(f"RAM memory % used: {psutil.virtual_memory()[2]}")


@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def load_model(model_name):
    return (
        AutoModelForSequenceClassification.from_pretrained(model_name),
        AutoTokenizer.from_pretrained(model_name),
    )


def main():

    st.title("Upload a CV and check if the canditate is relevant to the role..")

    models = {
        "sampathkethineedi/industry-classification": "DistilBERT Model to classify a business description into one of 62 industry tags.",
    }
    model_name = st.sidebar.selectbox(
        "Choose a classification model", list(models.keys())
    )
    model, tokenizer = load_model(model_name)
    if model_name.startswith("textattack/"):
        model.config.id2label = {0: "NEGATIVE (0) ", 1: "POSITIVE (1)"}
    model.eval()
    cls_explainer = SequenceClassificationExplainer(model=model, tokenizer=tokenizer)
    if cls_explainer.accepts_position_ids:
        emb_type_name = st.sidebar.selectbox(
            "Choose embedding type for attribution.", ["word", "position"]
        )
        if emb_type_name == "word":
            emb_type_num = 0
        if emb_type_name == "position":
            emb_type_num = 1
    else:
        emb_type_num = 0

    explanation_classes = ["predicted"] + list(model.config.label2id.keys())
    explanation_class_choice = st.sidebar.selectbox(
        "Explanation class: The class you would like to explain output with respect to.",
        explanation_classes,
    )
    # my_expander = st.beta_expander(

    
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
        print_memory_usage()

        st.text("Output")
        with st.spinner("Interpreting your text (This may take some time)"):
            if explanation_class_choice != "predicted":
                word_attributions = cls_explainer(
                    info,
                    class_name=explanation_class_choice,
                    embedding_type=emb_type_num,
                    internal_batch_size=2,
                )
            else:
                word_attributions = cls_explainer(
                    info, embedding_type=emb_type_num, internal_batch_size=2
                )

        if word_attributions:
            word_attributions_expander = st.beta_expander(
                "Click here for raw word attributions"
            )
            with word_attributions_expander:
                st.json(word_attributions)
            components.v1.html(
                cls_explainer.visualize()._repr_html_(), scrolling=True, height=350
            )


if __name__ == "__main__":
    main()
