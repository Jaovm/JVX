import streamlit as st
from utils import macro, fundamental, optimizer, risk

st.set_page_config(page_title="Painel SPX Falcon", layout="wide")

st.title("Painel de Suporte à Decisão - SPX Falcon")
st.markdown("Este painel integra **análise fundamentalista, quantitativa e macroeconômica**, inspirado na gestão do SPX Falcon Master.")

menu = st.sidebar.selectbox(
    "Selecione uma opção:",
    ["Dashboard Macro", "Análise de Ações", "Otimização de Carteira", "Simulador de Aporte", "Gestão de Risco"]
)

if menu == "Dashboard Macro":
    macro.show_macro_dashboard()

elif menu == "Análise de Ações":
    fundamental.run_analysis()

elif menu == "Otimimização de Carteira":
    optimizer.optimize_portfolio()

elif menu == "Simulador de Aporte":
    optimizer.simulate_aporte()

elif menu == "Gestão de Risco":
    risk.show_risk_dashboard()
