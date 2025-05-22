import streamlit as st
import pandas as pd
import numpy as np

def dashboard_macro():
    st.title("Dashboard Macroeconômico")
    st.subheader("Indicadores Econômicos")

    st.markdown("**Exemplo de Indicadores:**")

    col1, col2, col3 = st.columns(3)

    col1.metric("IPCA (12M)", "4,2%", "+0,3%")
    col2.metric("Selic", "10,50%", "-0,25%")
    col3.metric("PIB (Último)", "2,3%", "+0,4%")

    st.subheader("Cenário Atual")
    st.info("Cenário Neutro, com viés levemente expansionista. Beneficia setores como Consumo Discricionário, Tecnologia e Agronegócio.")

    st.subheader("Boletim Focus (Exemplo Simulado)")
    data = {
        'Indicador': ['IPCA', 'Selic', 'PIB', 'Câmbio'],
        'Projeção': ['4,2%', '9,75%', '2,1%', '4,90']
    }
    df = pd.DataFrame(data)
    st.table(df)
