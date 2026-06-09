import os
import streamlit as st
import sqlite3
import pandas as pd
from router import route, classify_query
from config import config

st.set_page_config(page_title="Arabic Query Agent", layout="centered")

tab1, tab2 = st.tabs(["Agent", "Data"])

with tab1:
    st.title("🤖 Arabic Query Agent")
    query = st.text_input("اكتب سؤالك هنا")

    if st.button("ابحث") and query:
        with st.spinner("جاري المعالجة..."):
            decision = classify_query(query)
            answer = route(query)

        colors = {"SQL": "🟦 SQL", "RAG": "🟩 RAG", "HYBRID": "🟨 HYBRID"}
        st.markdown(f"**Route:** {colors.get(decision, decision)}")
        st.divider()
        st.markdown(f"**الإجابة:**\n\n{answer}")

with tab2:
    st.subheader("Employees Database")
    conn = sqlite3.connect(config["database"]["path"])
    df = pd.read_sql("SELECT * FROM employees", conn)
    conn.close()
    st.dataframe(df)

    st.subheader("Remote Work Policy")
    for filename in config["documents"]["files"]:
        path = os.path.join(config["documents"]["dir"], filename)
        with open(path, "r", encoding="utf-8") as f:
            st.text(f.read())