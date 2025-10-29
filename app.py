import pandas as pd
import streamlit as st
from transformers import pipeline
import plotly.express as px
from used_func import get_top_n_words_en, get_top_n_words_id, convert_for_download
import torch
import re, time


bahasa = st.selectbox("Choose Language: ", ["English", "Indonesia"])

if "last_lang" not in st.session_state:
    st.session_state.last_lang = bahasa

if bahasa != st.session_state.last_lang:
    st.session_state.clear()
    st.session_state.bahasa = bahasa
    st.rerun()


tab_file, tab_teks = st.tabs(["File", "Text"])

if bahasa == "English":
    nlp = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    with tab_file:
        file_table_en = st.file_uploader("Please upload yoru data here!", type=["csv", "xlsx"], key=f"data {bahasa}")
        if file_table_en is not None:
            
            if file_table_en.name.endswith(".csv"):
                df = pd.read_csv(file_table_en)
            elif file_table_en.name.endswith(".xlsx"):
                df = pd.read_excel(file_table_en)
                
            st.toast("📂 Data successfully loaded. Preparing for processing...")
            st.write("📊 Preview Data:")
            st.dataframe(df.head())
            
            if "Sentiment" not in df.columns:
                df['Sentiment'] = df['komentar'].apply(lambda x: nlp(x)[0]["label"])
            st.write("📊 Result:")
            st.dataframe(df.head())
            
            csv = convert_for_download(df)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sentiment_english.csv",
                mime="text/csv",
                icon=":material/download:",
            )
            
            data = df["Sentiment"].value_counts()
            fig = px.pie(data, names=data.index, values=data.values, title="How Many?")
            st.plotly_chart(fig, use_container_width=True)
            
            num = st.number_input("How Many K?", min_value = 1, max_value=10, value=5)
            sentiment = st.selectbox("Choose sentiment: ", ["positive", "negative", "neutral"])
            ngram = st.slider("Select a range of values", 1, 3, (1, 1))

            result = get_top_n_words_en(corpus = df[df["Sentiment"] == sentiment]["komentar"], n=num, ngram_range=ngram)
            result_df = pd.DataFrame(result, columns=["Word", "Jumlah"])
            st.dataframe(result_df)
            st.bar_chart(result_df, x="Word", y="Jumlah", x_label="Kata", y_label="Banyaknya kemunculan")
            
            
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
        if "messages_en" not in st.session_state:
            st.session_state.messages_en = []
        
        text = st.chat_input("Write Your Sentiment Here") 
        if text:
            st.session_state.messages_en.append({"role": "user", "content": text})
            
            sentiment = nlp(text)[0]["label"]    
            st.session_state.messages_en.append({"role": "ai", "content": sentiment})
            
            for msg in st.session_state.messages_en:
                # For each message, create a chat message bubble with the appropriate role ("user" or "assistant").
                with st.chat_message(msg["role"]):
                    # Display the content of the message using Markdown for nice formatting.
                    st.markdown(msg["content"]) 
        
        
        
        
elif bahasa == "Indonesia":
    nlp = pipeline("sentiment-analysis", model="w11wo/indonesian-roberta-base-sentiment-classifier")
    
    with tab_file:
        file_table_id = st.file_uploader("Please upload yoru data here!", type=["csv", "xlsx"], key=f"data {bahasa}")
        if file_table_id is not None:
            if file_table_id.name.endswith(".csv"):
                df = pd.read_csv(file_table_id)
            elif file_table_id.name.endswith(".xlsx"):
                df = pd.read_excel(file_table_id)
                
            st.toast("📂 Data successfully loaded. Preparing for processing...")
            st.write("📊 Preview Data:")
            st.dataframe(df.head())
            
            if "Sentiment" not in df.columns:
                df['Sentiment'] = df['komentar'].apply(lambda x: nlp(x)[0]["label"])
            st.write("📊 Result:")
            st.dataframe(df.head())
            
            csv = convert_for_download(df)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sentiment_indo.csv",
                mime="text/csv",
                icon=":material/download:",
            )
            
            data = df["Sentiment"].value_counts()
            fig = px.pie(data, names=data.index, values=data.values, title="How Many?")
            st.plotly_chart(fig, use_container_width=True)
            
            num = st.number_input("How Many K?", min_value = 1, max_value=10, value=5)
            sentiment = st.selectbox("Choose sentiment: ", ["positive", "negative", "neutral"])
            ngram = st.slider("Select a range of values", 1, 3, (1, 1))

            result = get_top_n_words_id(corpus = df[df["Sentiment"] == sentiment]["komentar"], n=num, ngram_range=ngram)
            result_df = pd.DataFrame(result, columns=["Word", "Jumlah"])
            st.dataframe(result_df)
            st.bar_chart(result_df, x="Word", y="Jumlah", x_label="Kata", y_label="Banyaknya kemunculan")
            
            
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
        if "messages_indo" not in st.session_state:
            st.session_state.messages_indo = []
        
        text = st.chat_input("Write Your Sentiment Here") 
        if text:
            st.session_state.messages_indo.append({"role": "user", "content": text})
            
            sentiment = nlp(text)[0]["label"]    
            st.session_state.messages_indo.append({"role": "ai", "content": sentiment})
            
            for msg in st.session_state.messages_indo:
                # For each message, create a chat message bubble with the appropriate role ("user" or "assistant").
                with st.chat_message(msg["role"]):
                    # Display the content of the message using Markdown for nice formatting.
                    st.markdown(msg["content"]) 
        
        

        



















