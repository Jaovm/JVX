import streamlit as st
import yfinance as yf
import pandas as pd

def analise_fundamental():
    st.title("Análise Fundamentalista + Quantitativa")

    tickers = st.text_input("Digite os tickers separados por espaço (Ex: ITUB3.SA PETR4.SA WEGE3.SA)").split()

    if tickers:
        st.subheader("Dados Fundamentais")
        for ticker in tickers:
            st.write(f"**{ticker}**")
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                st.markdown(f"""
                - **Setor:** {info.get('sector')}
                - **Indústria:** {info.get('industry')}
                - **Beta:** {info.get('beta')}
                - **Dividend Yield:** {round(info.get('dividendYield', 0) * 100, 2)}%
                - **P/L:** {info.get('trailingPE')}
                - **ROE:** {info.get('returnOnEquity')}
                - **Margem Líquida:** {info.get('profitMargins')}
                """)
            except:
                st.warning(f"Não foi possível obter dados para {ticker}")
