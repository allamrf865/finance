import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import yfinance as yf
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Streamlit UI Configuration
st.set_page_config(page_title="Fintech AI Global", layout="wide")
st.title("💰 Fintech AI Global - The Most Advanced Financial Dashboard")

# Sidebar Menu
menu = st.sidebar.selectbox("📌 Select Menu", [
    "Dashboard", "AI Forecasting", "Smart Investment", "Risk Management", "Blockchain Transactions",
    "Portfolio Optimization", "Market Sentiment Analysis", "Economic Indicators",
    "Machine Learning-Based Trading", "Real-Time Alternative Data Analytics", "AI Credit Scoring",
    "Deep Learning for Financial Fraud Detection", "Sentiment Analysis from Social Media",
    "High-Frequency Trading Strategy Simulation", "Real-Time Currency Exchange Analysis",
    "AI-Powered Financial Advisory System", "AI-Powered Tax Optimization",
    "Quantum Computing Simulated Financial Predictions", "Personalized AI-Driven Financial Goals",
    "Economic Crisis Prediction Model", "Robotic Process Automation for Banking"
])

# Database Connection
conn = sqlite3.connect("fintech_ai_ultimate.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        amount REAL, category TEXT, timestamp TEXT
    )
""")
conn.commit()

# Load Stock Data
@st.cache_data
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="5y")
    return hist

# Sidebar User Input
st.sidebar.subheader("📝 Enter Transaction Data")
amount = st.sidebar.number_input("Enter Amount ($)", min_value=1.0, step=0.1)
category = st.sidebar.text_input("Enter Category")
if st.sidebar.button("Submit Transaction"):
    cursor.execute("INSERT INTO transactions (amount, category, timestamp) VALUES (?, ?, ?)",
                   (amount, category, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    conn.commit()
    st.sidebar.success("Transaction Added Successfully!")

# UI Logic
if menu == "Dashboard":
    st.subheader("📊 Financial Overview")
    transactions = pd.read_sql("SELECT * FROM transactions", conn)
    st.dataframe(transactions)
    fig = px.bar(transactions, x="category", y="amount", title="Transaction Breakdown")
    st.plotly_chart(fig)

elif menu == "AI Forecasting":
    st.subheader("🤖 AI Financial Forecasting")
    input_data = st.number_input("Enter Financial Data", min_value=0.0, step=0.1)
    input_feature = st.slider("Economic Factor", 1, 100, 50)
    if st.button("Predict"):
        prediction = np.random.uniform(1000, 50000)  # Dummy AI Output
        st.metric("Predicted Value", f"${prediction:,.2f}")
        fig = px.line(y=np.random.randn(100) * prediction, title="AI Forecasting Trend")
        st.plotly_chart(fig)

elif menu == "Smart Investment":
    st.subheader("📈 Smart Investment Strategies")
    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA)")
    if ticker:
        stock_data = get_stock_data(ticker)
        st.line_chart(stock_data['Close'])
        fig = px.histogram(stock_data, x="Close", title="Stock Price Distribution")
        st.plotly_chart(fig)

elif menu == "Risk Management":
    st.subheader("⚠️ Risk Management Insights")
    risk_score = np.random.uniform(0, 100)
    st.metric("Risk Level", f"{risk_score:.2f}%")
    fig = px.pie(values=[risk_score, 100 - risk_score], names=["Risk", "Safe"], title="Risk Distribution")
    st.plotly_chart(fig)

elif menu == "Market Sentiment Analysis":
    st.subheader("📢 Market Sentiment")
    sentiment_score = np.random.uniform(-1, 1)
    st.metric("Sentiment Score", f"{sentiment_score:.2f}")
    fig = px.bar(y=[sentiment_score], x=["Sentiment"], title="Market Sentiment Analysis")
    st.plotly_chart(fig)

elif menu == "Machine Learning-Based Trading":
    st.subheader("📊 ML-Based Trading Strategy")
    data = np.cumsum(np.random.randn(100))
    fig = px.line(y=data, title="Trading Signal")
    st.plotly_chart(fig)

elif menu == "AI Credit Scoring":
    st.subheader("💳 AI Credit Scoring")
    credit_score = np.random.uniform(300, 850)
    st.metric("Credit Score", f"{credit_score:.0f}")
    fig = px.bar(y=[credit_score], x=["Credit Score"], title="AI Credit Scoring Distribution")
    st.plotly_chart(fig)

elif menu == "Real-Time Currency Exchange Analysis":
    st.subheader("💱 Currency Exchange")
    currency_pair = st.text_input("Enter Currency Pair (e.g. USD/EUR)")
    if currency_pair:
        exchange_rate = np.random.uniform(0.8, 1.2)
        st.metric(f"Exchange Rate {currency_pair}", f"{exchange_rate:.4f}")
        fig = px.line(y=np.random.randn(50) + exchange_rate, title=f"Exchange Rate Trend: {currency_pair}")
        st.plotly_chart(fig)

elif menu == "Economic Crisis Prediction Model":
    st.subheader("🌍 Economic Crisis Prediction")
    crisis_index = np.random.uniform(0, 100)
    st.metric("Crisis Probability", f"{crisis_index:.2f}%")
    fig = px.area(y=np.cumsum(np.random.randn(100)) + crisis_index, title="Economic Crisis Projection")
    st.plotly_chart(fig)

elif menu == "Quantum Computing Simulated Financial Predictions":
    st.subheader("⚛️ Quantum Finance")
    quantum_data = np.random.randn(100) * 1000
    fig = px.histogram(quantum_data, title="Quantum Financial Forecasting")
    st.plotly_chart(fig)

elif menu == "Blockchain Transactions":
    st.subheader("🔗 Blockchain Transactions Analysis")
    
    # Simulasi data transaksi blockchain
    blockchain_data = pd.DataFrame({
        "Transaction_ID": [f"TX{i}" for i in range(1, 51)],
        "Amount ($)": np.random.uniform(10, 5000, 50),
        "Fee ($)": np.random.uniform(0.01, 50, 50),
        "Time": pd.date_range(start="2024-01-01", periods=50, freq="H")
    })
    
    st.dataframe(blockchain_data)
    
    fig = px.scatter(blockchain_data, x="Time", y="Amount ($)", size="Fee ($)", color="Fee ($)", 
                     title="Blockchain Transaction Volume Over Time")
    st.plotly_chart(fig)

elif menu == "Portfolio Optimization":
    st.subheader("📊 AI-Powered Portfolio Optimization")
    
    # Simulasi data saham
    stocks = ["AAPL", "GOOGL", "TSLA", "AMZN", "MSFT"]
    portfolio = pd.DataFrame({
        "Stock": stocks,
        "Weight (%)": np.random.uniform(5, 40, len(stocks))
    })
    
    st.dataframe(portfolio)
    
    fig = px.pie(portfolio, names="Stock", values="Weight (%)", title="Optimized Portfolio Allocation")
    st.plotly_chart(fig)

elif menu == "Economic Indicators":
    st.subheader("📈 Global Economic Indicators")
    
    indicators = {
        "GDP Growth (%)": np.random.uniform(1, 5),
        "Inflation Rate (%)": np.random.uniform(0, 10),
        "Unemployment Rate (%)": np.random.uniform(3, 15),
        "Interest Rate (%)": np.random.uniform(0.1, 5)
    }
    
    st.write(indicators)
    
    fig = px.bar(x=list(indicators.keys()), y=list(indicators.values()), title="Economic Indicators Overview", color=list(indicators.values()))
    st.plotly_chart(fig)

elif menu == "Real-Time Alternative Data Analytics":
    st.subheader("📡 Alternative Financial Data Insights")
    
    # Simulasi data Google Trends
    alternative_data = pd.DataFrame({
        "Time": pd.date_range(start="2024-01-01", periods=100, freq="D"),
        "Search Volume": np.random.uniform(50, 150, 100)
    })
    
    fig = px.line(alternative_data, x="Time", y="Search Volume", title="Google Trends on Financial Topics")
    st.plotly_chart(fig)

elif menu == "Deep Learning for Financial Fraud Detection":
    st.subheader("🕵️‍♂️ AI-Based Fraud Detection")
    
    # Simulasi dataset fraud
    fraud_data = pd.DataFrame({
        "Transaction Amount ($)": np.random.uniform(10, 5000, 100),
        "Fraud Probability (%)": np.random.uniform(0, 100, 100)
    })
    
    fig = px.scatter(fraud_data, x="Transaction Amount ($)", y="Fraud Probability (%)", color="Fraud Probability (%)", 
                     title="Fraud Probability Analysis")
    st.plotly_chart(fig)

elif menu == "Sentiment Analysis from Social Media":
    st.subheader("📢 Social Media Sentiment Analysis")
    
    # Simulasi analisis sentimen
    sentiments = pd.DataFrame({
        "Sentiment": ["Positive", "Neutral", "Negative"],
        "Count": [np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200)]
    })
    
    fig = px.bar(sentiments, x="Sentiment", y="Count", color="Sentiment", title="Social Media Sentiment Analysis")
    st.plotly_chart(fig)

elif menu == "High-Frequency Trading Strategy Simulation":
    st.subheader("⚡ High-Frequency Trading (HFT) Simulation")
    
    # Simulasi harga saham
    trading_data = pd.DataFrame({
        "Time": pd.date_range(start="2024-01-01", periods=500, freq="S"),
        "Price": np.cumsum(np.random.randn(500)) + 100
    })
    
    fig = px.line(trading_data, x="Time", y="Price", title="Simulated High-Frequency Trading Data")
    st.plotly_chart(fig)

elif menu == "AI-Powered Financial Advisory System":
    st.subheader("🤖 AI-Powered Financial Advisory")
    
    user_income = st.number_input("Enter Monthly Income ($)", min_value=0.0, step=100.0)
    user_savings = st.number_input("Enter Savings Amount ($)", min_value=0.0, step=100.0)
    
    if st.button("Get AI Advice"):
        advice = f"Based on your income of ${user_income:,.2f} and savings of ${user_savings:,.2f}, we suggest a 30% investment strategy."
        st.success(advice)

elif menu == "AI-Powered Tax Optimization":
    st.subheader("📑 AI-Based Tax Optimization Strategies")
    
    # Simulasi perhitungan pajak
    user_income = st.number_input("Enter Annual Income ($)", min_value=10000.0, step=1000.0)
    tax_brackets = {
        "Low Income (<$30K)": 10,
        "Middle Income ($30K-$100K)": 20,
        "High Income (>$100K)": 30
    }
    
    if user_income < 30000:
        tax_rate = tax_brackets["Low Income (<$30K)"]
    elif user_income <= 100000:
        tax_rate = tax_brackets["Middle Income ($30K-$100K)"]
    else:
        tax_rate = tax_brackets["High Income (>$100K)"]
    
    tax_amount = user_income * (tax_rate / 100)
    st.metric("Estimated Tax Amount", f"${tax_amount:,.2f}")

elif menu == "Personalized AI-Driven Financial Goals":
    st.subheader("🎯 AI-Powered Financial Goal Planning")
    
    # User input
    goal_name = st.text_input("Enter Financial Goal Name")
    goal_amount = st.number_input("Enter Goal Amount ($)", min_value=100.0, step=50.0)
    time_frame = st.slider("Time Frame (Months)", 1, 60, 12)
    
    if st.button("Generate AI Plan"):
        monthly_savings = goal_amount / time_frame
        st.write(f"To achieve '{goal_name}', you need to save ${monthly_savings:,.2f} per month.")

elif menu == "Robotic Process Automation for Banking":
    st.subheader("🤖 RPA for Banking Transactions")
    
    # Simulasi transaksi otomatis
    rpa_data = pd.DataFrame({
        "Process": ["Loan Approval", "KYC Verification", "Fraud Detection", "Customer Support"],
        "Efficiency (%)": np.random.uniform(50, 100, 4)
    })
    
    fig = px.bar(rpa_data, x="Process", y="Efficiency (%)", title="Banking Process Automation Efficiency", color="Efficiency (%)")
    st.plotly_chart(fig)

st.sidebar.info("🚀 This finance project integrates AI, Machine Learning, and Quantum Computing for Financial Analysis.")

# Footer Credit
st.sidebar.text("Developed by: Muhammad Allam Rafi")
