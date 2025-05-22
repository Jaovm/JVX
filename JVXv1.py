import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Função para obter dados
@st.cache_data
def get_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data.dropna()

# Layout do App
st.set_page_config(page_title="Painel de Ações - Estilo SPX", layout="wide")

st.title("Painel de Suporte à Decisão para Ações Brasileiras")
st.subheader("Inspirado na Filosofia de Gestão do SPX Falcon Master")

# Input de Tickers
tickers = st.multiselect(
    "Selecione os Tickers (Ex: ITUB3.SA, PETR4.SA, VALE3.SA)",
    ['ITUB3.SA', 'PETR4.SA', 'VALE3.SA', 'WEGE3.SA', 'BBAS3.SA', 'PRIO3.SA', 'EGIE3.SA'],
    default=['ITUB3.SA', 'VALE3.SA', 'WEGE3.SA']
)

# Input de Período
start_date = st.date_input("Data Inicial", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("Data Final", value=pd.to_datetime("today"))

# Carregar Dados
if tickers:
    data = get_data(tickers, start_date, end_date)

    st.subheader("Preço Ajustado")
    st.line_chart(data)

    # Retornos
    returns = data.pct_change().dropna()

    st.subheader("Matriz de Correlação")
    corr = returns.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(plt.gcf())
    plt.clf()

    # Análise Quantitativa
    st.subheader("Análise Quantitativa")

    volatility = returns.std() * np.sqrt(252)
    cumulative_return = (1 + returns).cumprod().iloc[-1] - 1
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

    metrics = pd.DataFrame({
        "Volatilidade Anualizada": volatility,
        "Retorno Acumulado": cumulative_return,
        "Sharpe": sharpe_ratio
    }).sort_values("Sharpe", ascending=False)

    st.dataframe(metrics.style.format({
        "Volatilidade Anualizada": "{:.2%}",
        "Retorno Acumulado": "{:.2%}",
        "Sharpe": "{:.2f}"
    }))

    # Momentum (12 meses)
    st.subheader("Indicador de Momentum (12 Meses)")

    momentum = data.pct_change(252).dropna()
    st.bar_chart(momentum.iloc[-1])

    # Z-Score de Valuation Simples (Preço vs. Média 200)
    st.subheader("Z-Score vs. Média Móvel 200 dias")

    z_scores = {}
    for ticker in tickers:
        price = data[ticker]
        mean = price.rolling(window=200).mean()
        std = price.rolling(window=200).std()
        z = (price.iloc[-1] - mean.iloc[-1]) / std.iloc[-1] if std.iloc[-1] != 0 else np.nan
        z_scores[ticker] = z

    z_df = pd.DataFrame.from_dict(z_scores, orient='index', columns=['Z-Score'])
    st.dataframe(z_df.style.format({"Z-Score": "{:.2f}"}))

    # Simulação de Stress Test (Queda de 5%)
    st.subheader("Stress Test - Queda Simulada de 5% nos Ativos")

    stress_return = returns.mean() - 0.05
    stress_pnl = stress_return * 1000000  # Simulando uma carteira de R$ 1 milhão

    stress_df = pd.DataFrame({
        'Stress Return (%)': stress_return * 100,
        'Perda Estimada (R$)': stress_pnl
    })

    st.dataframe(stress_df.style.format({
        'Stress Return (%)': '{:.2f}%',
        'Perda Estimada (R$)': 'R$ {:,.2f}'
    }))

    # Distribuição dos Retornos
    st.subheader("Distribuição dos Retornos Diários")
    fig, ax = plt.subplots()
    returns.plot.hist(bins=50, alpha=0.7, ax=ax)
    st.pyplot(fig)

else:
    st.warning("Selecione pelo menos um ticker para iniciar a análise.")
