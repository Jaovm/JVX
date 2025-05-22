import streamlit as st
import yfinance as yf
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns

def otimizacao_carteira():
    st.title("Otimização de Carteira - SPX Falcon")

    tickers = st.text_input("Tickers separados por espaço (Ex: WEGE3.SA PETR4.SA ITUB4.SA)").split()

    if tickers:
        st.subheader("Período de Dados")
        periodo = st.selectbox("Período", ["1y", "3y", "5y"])

        st.subheader("Dados de Preços")
        dados = yf.download(tickers, period=periodo)['Adj Close']
        st.line_chart(dados)

        st.subheader("Otimização")
        retornos = expected_returns.mean_historical_return(dados)
        matriz_risco = risk_models.sample_cov(dados)

        ef = EfficientFrontier(retornos, matriz_risco)
        pesos = ef.max_sharpe()
        limpos = ef.clean_weights()

        st.write("Pesos Ótimos para Maior Sharpe:")
        st.json(limpos)

        performance = ef.portfolio_performance(verbose=True)
