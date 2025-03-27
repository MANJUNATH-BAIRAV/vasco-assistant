import datetime
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import io
from PIL import Image
import pytesseract
from langchain_community.llms import Ollama

# Initialize AI model
llm = Ollama(model="llama2")  

# Read transactions data
df = pd.read_csv('transactions_2022_2023_categorized.csv')

# Process data
df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month'] = pd.to_datetime(df['Date']).dt.month
df['Month Name'] = pd.to_datetime(df['Date']).dt.strftime("%b")
df = df.drop(columns=['Transaction', 'Transaction vs category'])
df['Category'] = np.where(df['Expense/Income'] == 'Income', df['Name / Description'], df['Category'])

# Streamlit Sidebar
selected_feature = st.sidebar.selectbox("Select Feature", ["AI Finance Chat", "Financial Report Analysis", "Expense Dashboard"])

# AI Finance Chat
if selected_feature == "AI Finance Chat":
    st.title("Vasco Assistant - AI Finance Chat")
    session_state = st.session_state
    if 'chat_history' not in session_state:
        session_state.chat_history = []
    
    user_prompt = st.text_input("Ask a question:", key="user_input")
    
    if st.button("Submit", key="chat_submit"):
        if user_prompt:
            result = llm.predict(user_prompt + " Give a short reply")
            session_state.chat_history.append((user_prompt, result))
    
    st.subheader("Chat History:")
    for question, response in reversed(session_state.chat_history):
        st.write(f"**You:** {question}")
        st.write(f"**Vasco Assistant:** {response}")
        st.write("---")
    
    if st.sidebar.button("Clear Chat History", key="clear_chat"):
        session_state.chat_history = []
        st.success("Chat history cleared!")

# Financial Report Analysis
elif selected_feature == "Financial Report Analysis":
    st.title("Vasco Assistant - Financial Report Analysis")
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], key="image_upload")
    
    if uploaded_image:
        if st.button("Perform OCR & Analyze", key="ocr_analyze_button"):
            img = Image.open(io.BytesIO(uploaded_image.read()))
            ocr_result = pytesseract.image_to_string(img)
            llm_result = llm.predict(ocr_result + " Summarize the financial information.")
            
            st.image(img, caption="Uploaded Image")
            st.subheader("OCR Extracted Text:")
            st.text(ocr_result)
            st.subheader("AI Model's Financial Analysis:")
            st.text(llm_result)
    else:
        st.warning("Please upload an image first.")

# Expense Dashboard
elif selected_feature == "Expense Dashboard":
    st.title("Vasco Assistant - Expense Dashboard")
    year_selection = st.sidebar.radio("Select Year", ['2022', '2023'], key="year_select")
    
    def make_pie_chart(year, label):
        sub_df = df[(df['Expense/Income'] == label) & (df['Year'] == year)]
        color_scale = px.colors.qualitative.Set1  # Vibrant color set
        pie_fig = px.pie(sub_df, values='Amount (EUR)', names='Category', color_discrete_sequence=color_scale)
        pie_fig.update_traces(textposition='inside', hole=0.3, textinfo="label+percent")
        return pie_fig
    
    def make_monthly_bar_chart(year, label):
        sub_df = df[(df['Expense/Income'] == label) & (df['Year'] == year)]
        total_by_month = sub_df.groupby(['Month', 'Month Name'])['Amount (EUR)'].sum().reset_index()
        color_scale = px.colors.sequential.Teal if label == "Income" else px.colors.sequential.Magma  
        return px.bar(total_by_month, x='Month Name', y='Amount (EUR)', color='Amount (EUR)', color_continuous_scale=color_scale)
    
    st.plotly_chart(make_pie_chart(int(year_selection), 'Expense'), use_container_width=True)
    st.plotly_chart(make_monthly_bar_chart(int(year_selection), 'Income'), use_container_width=True)
    st.plotly_chart(make_monthly_bar_chart(int(year_selection), 'Expense'), use_container_width=True)
