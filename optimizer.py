import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from riskfolio import Portfolio
import matplotlib.pyplot as plt

def load_returns(tickers):
    data = yf.download(tickers, period="3y")['Adj Close']
    returns = data.pct_change().dropna()
    return returns

def hrp_optimization(returns):
    p = Portfolio(returns=returns)
    p.assets_stats(method_mu='hist', method_cov='ledoit')
    p.optimize(model='HRP')
    weights = p.weights.T
    return weights

def markowitz_optimization(returns):
    p = Portfolio(returns=returns)
    p.assets_stats(method_mu='hist', method_cov='ledoit')
    p.optimize(model='MaxSharpe', risk_free_rate=0.11/12)
    weights = p.weights.T
    return weights

def optimize_portfolio():
    st.subheader("Otimização de Carteira")

    tickers = st.multiselect("Selecione os ativos:", ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'WEGE3.SA'])
    if len(tickers) < 2:
        st.warning("Selecione pelo menos 2 ativos.")
        return

    returns = load_returns(tickers)

    method = st.radio("Método de Otimização", ["HRP", "Fronteira Eficiente"])

    if method == "HRP":
        weights = hrp_optimization(returns)
    else:
        weights = markowitz_optimization(returns)

    st.dataframe(weights.style.format("{:.2%}"))

    fig, ax = plt.subplots()
    weights.plot.pie(y=weights.columns[0], autopct='%1.1f%%', ax=ax)
    plt.ylabel("")
    st.pyplot(fig)

def simulate_aporte():
    st.subheader("Simulador de Aporte")
    st.info("Funcionalidade em desenvolvimento.")