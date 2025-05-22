import streamlit as st

def show_macro_dashboard():
    st.subheader("Dashboard Macroeconômico")

    st.metric("Selic", "10,25%", "-0,50pp")
    st.metric("IPCA 12M", "3,9%", "-0,1pp")
    st.metric("PIB Projeção", "2,1%", "+0,2pp")
    st.metric("Dólar", "5,10", "-1,5%")
    st.metric("Petróleo (Brent)", "85 USD", "+0,8%")

    st.info("**Cenário Classificado como: Neutro**")
    st.success("Setores Favorecidos: Saúde, Bancos, Seguradoras, Utilities")

    st.markdown("A classificação automática é baseada nas projeções de inflação, PIB e política monetária.")