"""
Project: Review Sentiment Analyzer
Author: Silvio Christian, Joe
Description:
    This Streamlit application performs sentiment analysis on English and Indonesian text.
    It leverages Hugging Face Transformer models (RoBERTa) for high-accuracy classification.
    Features include batch processing (CSV/Excel), interactive visualization (Plotly),
    and N-gram analysis for insight extraction.
"""

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from transformers import pipeline
import plotly.express as px
from utils import get_top_n_words_en, get_top_n_words_id, convert_for_download
from lime.lime_text import LimeTextExplainer
import torch
import re, time

# ==========================================
# 1. Language Selection & Session Management
# ==========================================
bahasa = st.selectbox("Choose Language: ", ["English", "Indonesia"])

# Initialize session state to track language changes
if "last_lang" not in st.session_state:
    st.session_state.last_lang = bahasa

# Detect if language has changed. If yes, clear memory/cache and reload UI.
if bahasa != st.session_state.last_lang:
    st.session_state.clear()
    st.session_state.bahasa = bahasa
    st.rerun()

# ==========================================
# 2. Main UI Layout (Tabs)
# ==========================================
tab_file, tab_teks = st.tabs(["File", "Text"])

# ==========================================
# 3. Logic: English Analysis
# ==========================================
if bahasa == "English":
    # Load Pre-trained Model: RoBERTa optimized for Sentiment Analysis
    nlp = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    with tab_file:
        # File Uploader Widget
        file_table_en = st.file_uploader("Please upload yoru data here!", type=["csv", "xlsx"], key=f"data {bahasa}")
        
        if file_table_en is not None:
            
            # Data Loading: Handle both CSV and Excel formats
            if file_table_en.name.endswith(".csv"):
                df = pd.read_csv(file_table_en)
            elif file_table_en.name.endswith(".xlsx"):
                df = pd.read_excel(file_table_en)
                
            st.toast("📂 Data successfully loaded. Preparing for processing...")
            st.write("📊 Preview Data:")
            st.dataframe(df.head())
            
            # Batch Inference: Apply model to the entire column if not already processed
            if "Sentiment" not in df.columns:
                df['Sentiment'] = df['komentar'].apply(lambda x: nlp(x)[0]["label"])

            if "Confidence" not in df.columns:
                df['Confidence'] = df['komentar'].apply(lambda x: nlp(x)[0]["score"])
            
            st.write("📊 Result:")
            st.dataframe(df.head())
            
            # Prepare CSV for Download
            csv = convert_for_download(df)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sentiment_english.csv",
                mime="text/csv",
                icon=":material/download:",
            )
            
            # Visualization: Sentiment Distribution Pie Chart
            data = df["Sentiment"].value_counts()
            fig = px.pie(data, names=data.index, values=data.values, title="How Many?")
            st.plotly_chart(fig, use_container_width=True)
            
            # N-Gram Analysis (Word Frequency)
            num = st.number_input("How Many K?", min_value = 1, max_value=10, value=5)
            sentiment = st.selectbox("Choose sentiment: ", ["positive", "negative", "neutral"])
            ngram = st.slider("Select a range of values", 1, 3, (1, 1))

            # Helper function call to extract top keywords
            result = get_top_n_words_en(corpus = df[df["Sentiment"] == sentiment]["komentar"], n=num, ngram_range=ngram)
            result_df = pd.DataFrame(result, columns=["Word", "Jumlah"])
            st.dataframe(result_df)
            st.bar_chart(result_df, x="Word", y="Jumlah", x_label="Kata", y_label="Banyaknya kemunculan")
            
            # Feature Engineering: Analyze Text Complexity (Sentence & Word Length)
            df['Panjang Kalimat'] = df["komentar"].apply(lambda x: len([x for x in re.split(r'[.!?]+', x) if x.strip()]))
            df['Panjang Kata'] = df["komentar"].apply(lambda x: len(x.split()))
            
            # Statistical Aggregation
            data_kalimat = df.groupby("Sentiment")["Panjang Kalimat"].mean().sort_values().reset_index()
            st.dataframe(data_kalimat)
            
            data_kata = df.groupby("Sentiment")["Panjang Kata"].mean().sort_values().reset_index()
            st.dataframe(data_kata)
            
            st.toast("📊 Data processing completed successfully.")
        else:
            st.write("⚠️ Belum ada file yang di-upload")
            
    with tab_teks:

        text = st.text_area("Write Your Sentiment Here")
        if st.button("Send"): 
            if text:
                # Perform Single Inference
                sentiment = nlp(text)[0]["label"]     
                conf = nlp(text)[0]["score"]     
                st.info(f"""
                    **Sentiment:** {sentiment}  
                    **Confidence:** {conf:.2f}
                """)

                def predict_function(texts):
                    if isinstance(texts, np.ndarray):
                        texts = texts.tolist()
                    if isinstance(texts, str):
                        texts = [texts]

                    validated_text = []
                    for text in texts:
                        if not text or text == "":
                            validated_text.append(".")
                        else:
                            validated_text.append(text)

                    predictions = nlp(validated_text, top_k=None) 

                    scores = []
                    for prediction in predictions:
                        sorted_pred = sorted(prediction, key=lambda x: x['label'])
                        
                        items = [item['score'] for item in sorted_pred]
                        scores.append(items)

                    return np.array(scores)

                with st.spinner("Analyzing with LIME..."):
                    explainer = LimeTextExplainer(class_names=['negative', 'neutral', 'positive'])
                    exp = explainer.explain_instance(
                        text_instance=text,
                        classifier_fn=predict_function,
                        num_features=5 # Top 5 Words
                    )
                    html_data = exp.as_html()
                    components.html(html_data, height=800, scrolling=True)
        
# ==========================================
# 4. Logic: Indonesian Analysis
# ==========================================
elif bahasa == "Indonesia":
    # Load Pre-trained Model: Indo-RoBERTa for Indonesian Sentiment
    nlp = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")
    
    with tab_file:
        # File Uploader for Indonesian Data
        file_table_id = st.file_uploader("Please upload yoru data here!", type=["csv", "xlsx"], key=f"data {bahasa}")
        if file_table_id is not None:
            if file_table_id.name.endswith(".csv"):
                df = pd.read_csv(file_table_id)
            elif file_table_id.name.endswith(".xlsx"):
                df = pd.read_excel(file_table_id)
                
            st.toast("📂 Data successfully loaded. Preparing for processing...")
            st.write("📊 Preview Data:")
            st.dataframe(df.head())
            
            # Apply Inference Row-by-Row
            if "Sentiment" not in df.columns:
                df['Sentiment'] = df['komentar'].apply(lambda x: nlp(x)[0]["label"])

            if "Confidence" not in df.columns:
                df['Confidence'] = df['komentar'].apply(lambda x: nlp(x)[0]["score"])
            
            st.write("📊 Result:")
            st.dataframe(df.head())
            
            # Prepare Download
            csv = convert_for_download(df)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sentiment_indo.csv",
                mime="text/csv",
                icon=":material/download:",
            )
            
            # Visualization: Pie Chart
            data = df["Sentiment"].value_counts()
            fig = px.pie(data, names=data.index, values=data.values, title="How Many?")
            st.plotly_chart(fig, use_container_width=True)
            
            # N-Gram Analysis Configuration
            num = st.number_input("How Many K?", min_value = 1, max_value=10, value=5)
            sentiment = st.selectbox("Choose sentiment: ", ["positive", "negative", "neutral"])
            ngram = st.slider("Select a range of values", 1, 3, (1, 1))

            # Extract Top Words (Indonesian Helper Function)
            result = get_top_n_words_id(corpus = df[df["Sentiment"] == sentiment]["komentar"], n=num, ngram_range=ngram)
            result_df = pd.DataFrame(result, columns=["Word", "Jumlah"])
            st.dataframe(result_df)
            st.bar_chart(result_df, x="Word", y="Jumlah", x_label="Kata", y_label="Banyaknya kemunculan")
            
            
            # Feature Engineering: Text Statistics
            df['Panjang Kalimat'] = df["komentar"].apply(lambda x: len([x for x in re.split(r'[.!?]+', x) if x.strip()]))
            df['Panjang Kata'] = df["komentar"].apply(lambda x: len(x.split()))
            
            data_kalimat = df.groupby("Sentiment")["Panjang Kalimat"].mean().sort_values().reset_index()
            st.dataframe(data_kalimat)
            
            data_kata = df.groupby("Sentiment")["Panjang Kata"].mean().sort_values().reset_index()
            st.dataframe(data_kata)
            
            st.toast("📊 Data processing completed successfully.")
        else:
            st.write("⚠️ Belum ada file yang di-upload")
        
    with tab_teks:
        # Chat Interface for Indonesian
        if "messages_indo" not in st.session_state:
            st.session_state.messages_indo = []
        
        text = st.text_area("Write Your Sentiment Here") 

        if st.button("Send"):
            if text:
                # Perform Single Inference
                sentiment = nlp(text)[0]["label"]     
                conf = nlp(text)[0]["score"]     
                st.info(f"""
                    **Sentiment:** {sentiment}  
                    **Confidence:** {conf:.2f}
                """)
            
                def predict_function(text):
                    if isinstance(texts, np.ndarray):
                        texts = texts.tolist()
                    if isinstance(texts, str):
                        texts = [texts]

                    validated_text = []
                    for text in texts:
                        if not text or text == "":
                            validated_text.append(".")
                        else:
                            validated_text.append(text)

                    predictions = nlp(validated_text, top_k=None) 

                    scores = []
                    for prediction in predictions:
                        sorted_pred = sorted(prediction, key=lambda x: x['label'])
                        
                        items = [item['score'] for item in sorted_pred]
                        scores.append(items)

                    return np.array(scores)

                with st.spinner("Analyzing with LIME..."):
                    explainer = LimeTextExplainer(class_names=['negative', 'neutral', 'positive'])
                    exp = explainer.explain_instance(
                        text_instance=text,
                        classifier_fn=predict_function,
                        num_features=5 # Top 5 Words
                    )
                    html_data = exp.as_html()
                    components.html(html_data, height=800, scrolling=True)
