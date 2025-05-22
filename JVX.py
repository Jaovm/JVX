import streamlit as st
import macro, fundamental, optimizer, risk

st.set_page_config(page_title="SPX Falcon - Painel de Investimentos", layout="wide")

st.sidebar.title("SPX Falcon - Painel")
menu = st.sidebar.selectbox("Menu", ["Dashboard Macro", "Análise Fundamentalista", "Otimização de Carteira", "Gestão de Riscos"])

if menu == "Dashboard Macro":
    macro.dashboard_macro()

elif menu == "Análise Fundamentalista":
    fundamental.analise_fundamental()

elif menu == "Otimização de Carteira":
    optimizer.otimizacao_carteira()

elif menu == "Gestão de Riscos":
    risk.monitoramento_riscos()
