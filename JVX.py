# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import yfinance as yf # Import yfinance

# Configuração da página
st.set_page_config(
    page_title="Análise de Ações Brasileiras - Modelo SPX FALCON (yfinance)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Lógica de Coleta e Processamento de Dados (Usando yfinance) ---

# Lista de ações (pode ser configurável no futuro)
acoes_default = [
    'AGRO3.SA', 'BBAS3.SA', 'BBSE3.SA', 'BPAC11.SA', 'EGIE3.SA', 'ITUB3.SA',
    'PRIO3.SA', 'PSSA3.SA', 'SAPR3.SA', 'SBSP3.SA', 'VIVT3.SA', 'WEGE3.SA',
    'TOTS3.SA', 'B3SA3.SA', 'TAEE3.SA', 'CMIG3.SA',
    'PETR4.SA', 'VALE3.SA', 'ITSA4.SA', 'ABEV3.SA', 'RENT3.SA', 'MGLU3.SA',
    'SUZB3.SA', 'RADL3.SA', 'CSAN3.SA', 'EQTL3.SA'
]

# Definir parâmetros setoriais para valuation (mantidos para lógica de cálculo)
setores = {
    'Financeiro': {
        'acoes': ['BBAS3.SA', 'ITUB3.SA', 'B3SA3.SA', 'BPAC11.SA', 'BBSE3.SA', 'PSSA3.SA', 'ITSA4.SA'],
        'metodo': 'Gordon Growth', # Simplificado para P/L ou P/VP heurístico
        'parametros': {'p_l_justo': 10, 'p_vp_justo': 1.5} # Exemplo de parâmetros
    },
    'Utilities': {
        'acoes': ['EGIE3.SA', 'CMIG3.SA', 'SBSP3.SA', 'SAPR3.SA', 'TAEE3.SA', 'EQTL3.SA'],
        'metodo': 'Fluxo de Caixa Descontado', # Simplificado para Dividend Yield ou P/L
        'parametros': {'dividend_yield_justo': 0.06, 'p_l_justo': 12}
    },
    'Tecnologia': {
        'acoes': ['TOTS3.SA', 'WEGE3.SA'],
        'metodo': 'Múltiplos de Receita', # Simplificado para P/L
        'parametros': {'p_l_justo': 20}
    },
    'Commodities': {
        'acoes': ['AGRO3.SA', 'PRIO3.SA', 'VALE3.SA', 'SUZB3.SA', 'CSAN3.SA', 'PETR4.SA'],
        'metodo': 'Ciclo de Preços', # Simplificado para P/L ou EV/EBITDA heurístico
        'parametros': {'p_l_justo': 8, 'ev_ebitda_justo': 6}
    },
    'Consumo': {
        'acoes': ['VIVT3.SA', 'ABEV3.SA', 'RENT3.SA', 'MGLU3.SA', 'RADL3.SA'],
        'metodo': 'Múltiplos Comparáveis', # Simplificado para P/L
        'parametros': {'p_l_justo': 15}
    }
}

# Função para buscar dados usando yfinance
@st.cache_data(ttl=1800) # Cache de 30 minutos para dados yfinance
def buscar_dados_yfinance(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Histórico de 5 anos, intervalo mensal
        hist = ticker.history(period="5y", interval="1mo")
        # Informações da empresa
        info = ticker.info
        # Recomendações (pode não estar sempre disponível)
        recom = ticker.recommendations
        return hist, info, recom
    except Exception as e:
        st.warning(f"Erro ao buscar dados para {ticker_symbol} via yfinance: {e}")
        return pd.DataFrame(), {}, pd.DataFrame()

# Função para processar os dados brutos do yfinance
def processar_dados_acao(simbolo, hist, info, recom):
    if hist.empty or not info:
        st.warning(f"Dados insuficientes do yfinance para {simbolo}. Pulando.")
        return None

    # Processar informações básicas
    preco_atual = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
    if preco_atual is None and not hist.empty:
        preco_atual = hist['Close'].iloc[-1] # Usa último fechamento se preço atual não disponível

    if preco_atual is None:
        st.warning(f"Não foi possível determinar o preço atual para {simbolo}. Pulando.")
        return None

    info_processada = {
        'symbol': info.get('symbol', simbolo),
        'name': info.get('shortName') or info.get('longName', 'N/A'),
        'currency': info.get('currency', 'BRL'),
        'exchange': info.get('exchange', 'SAO'),
        'current_price': preco_atual,
        '52w_high': info.get('fiftyTwoWeekHigh'),
        '52w_low': info.get('fiftyTwoWeekLow'),
        'sector': info.get('sector', 'N/A'), # Usar setor do yfinance se disponível
        'industry': info.get('industry', 'N/A'),
        'marketCap': info.get('marketCap'),
        'beta': info.get('beta'),
        'trailingPE': info.get('trailingPE'),
        'forwardPE': info.get('forwardPE'),
        'dividendYield': info.get('dividendYield')
    }

    # Processar recomendações (simplificado)
    # Pega a recomendação mais recente se disponível
    rating = 'N/A'
    if recom is not None and not recom.empty:
        # Ordena por data e pega a última
        recom_sorted = recom.sort_index()
        if not recom_sorted.empty:
            last_recom = recom_sorted.iloc[-1]
            rating = last_recom.get('To Grade') or last_recom.get('Action') # 'To Grade' é mais comum
            # Simplificar ratings comuns
            rating_lower = str(rating).lower()
            if 'buy' in rating_lower: rating = 'Compra'
            elif 'sell' in rating_lower: rating = 'Venda'
            elif 'hold' in rating_lower or 'neutral' in rating_lower: rating = 'Neutro'
            elif 'outperform' in rating_lower: rating = 'Compra'
            elif 'underperform' in rating_lower: rating = 'Venda Parcial'
            # Mantém outros ratings como estão

    insights_simplificados = {
        'recommendation_rating': rating, # Usaremos isso em vez dos scores técnicos
        'target_price_mean': info.get('targetMeanPrice'),
        'analyst_count': info.get('numberOfAnalystOpinions')
    }

    # Calcular valuation (usando a função adaptada)
    valuation = calcular_preco_justo_yfinance(simbolo, info_processada, insights_simplificados)

    # Renomear colunas do histórico para consistência (se necessário)
    hist.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    # Remover colunas extras que não usamos ('Dividends', 'Stock Splits')
    hist = hist[['open', 'high', 'low', 'close', 'volume']]

    return {
        'historico': df_to_serializable(hist),
        'info': info_processada,
        'insights': insights_simplificados,
        'valuation': valuation
    }

# Função para calcular preço justo (adaptada para dados yfinance)
def calcular_preco_justo_yfinance(simbolo, info, insights):
    preco_atual = info.get('current_price')
    setor_yf = info.get('sector') # Tenta usar o setor do yfinance
    beta = info.get('beta')
    pe_ratio = info.get('trailingPE')
    dividend_yield = info.get('dividendYield')

    setor_acao = None
    parametros = None
    metodo = None

    # Tenta mapear setor yfinance para nossa classificação
    # (Pode precisar de mapeamento mais robusto)
    if setor_yf:
        if setor_yf in ['Financial Services', 'Financials']: setor_acao = 'Financeiro'
        elif setor_yf in ['Utilities']: setor_acao = 'Utilities'
        elif setor_yf in ['Technology']: setor_acao = 'Tecnologia'
        elif setor_yf in ['Basic Materials', 'Energy']: setor_acao = 'Commodities'
        elif setor_yf in ['Consumer Cyclical', 'Consumer Defensive', 'Communication Services', 'Healthcare', 'Industrials', 'Real Estate']: setor_acao = 'Consumo' # Agrupando outros

    # Se não mapeou ou não tinha setor yf, tenta pela nossa lista
    if not setor_acao:
        for nome_setor, setor_info in setores.items():
            if simbolo in setor_info['acoes']:
                setor_acao = nome_setor
                break

    # Se ainda não encontrou, usa 'Geral'
    if not setor_acao:
        setor_acao = 'Geral'
        parametros = {'p_l_justo': 12} # Parâmetro genérico
        metodo = 'Múltiplos Gerais'
    else:
        parametros = setores[setor_acao]['parametros']
        metodo = setores[setor_acao]['metodo']

    # Tratar caso de preco_atual ser None ou inválido
    if preco_atual is None or not isinstance(preco_atual, (int, float)) or preco_atual <= 0:
        st.warning(f"Preço atual inválido ({preco_atual}) para {simbolo}. Não é possível calcular valuation.")
        return {
            'preco_atual': preco_atual,
            'preco_justo': np.nan,
            'preco_compra_forte': np.nan,
            'recomendacao': 'Indefinido',
            'setor': setor_acao,
            'metodo_valuation': metodo,
            'ajuste_tecnico': 1.0
        }

    # Ajuste simplificado baseado na recomendação de analistas (se houver)
    ajuste = 1.0
    rating = insights.get('recommendation_rating', 'N/A')
    if rating == 'Compra': ajuste = 1.05
    elif rating == 'Venda': ajuste = 0.95
    # (Pode refinar mais: Compra Forte -> 1.10, Venda Forte -> 0.90 etc)

    # Calcular preço justo baseado no método do setor (com heurísticas)
    preco_justo = np.nan
    try:
        # Prioriza P/L se disponível
        if pe_ratio and pd.notna(pe_ratio) and pe_ratio > 0 and 'p_l_justo' in parametros:
            preco_justo = preco_atual * (parametros['p_l_justo'] / pe_ratio) * ajuste
        # Tenta usar Dividend Yield para Utilities
        elif setor_acao == 'Utilities' and dividend_yield and pd.notna(dividend_yield) and dividend_yield > 0 and 'dividend_yield_justo' in parametros:
             preco_justo = preco_atual * (dividend_yield / parametros['dividend_yield_justo']) * ajuste
        # Fallback para P/L genérico se P/L específico não disponível
        elif 'p_l_justo' in parametros:
            preco_justo = preco_atual * (parametros['p_l_justo'] / 12.0) * ajuste # Usa 12 como P/L médio mercado proxy
        else:
             preco_justo = preco_atual * ajuste # Se nenhum método se aplica, usa só ajuste

        # Limita o preço justo para não ser muito distante do atual (ex: +/- 50%)
        if pd.notna(preco_justo):
            preco_justo = max(preco_atual * 0.5, min(preco_atual * 1.5, preco_justo))

    except Exception as e:
        st.warning(f"Erro no cálculo do preço justo para {simbolo} ({metodo}): {e}")
        preco_justo = np.nan

    # Calcular preço de compra forte
    preco_compra_forte = preco_justo * 0.8 if pd.notna(preco_justo) else np.nan

    # Determinar recomendação (baseada na nossa valuation)
    recomendacao_calculada = 'Indefinido'
    if pd.notna(preco_justo) and preco_atual > 0:
        if preco_atual < preco_compra_forte:
            recomendacao_calculada = "Compra Forte"
        elif preco_atual < preco_justo * 0.95:
            recomendacao_calculada = "Compra"
        elif preco_atual > preco_justo * 1.15:
            recomendacao_calculada = "Venda"
        elif preco_atual > preco_justo * 1.05:
            recomendacao_calculada = "Venda Parcial"
        else:
            recomendacao_calculada = "Neutro"

    return {
        'preco_atual': preco_atual,
        'preco_justo': preco_justo,
        'preco_compra_forte': preco_compra_forte,
        'recomendacao': recomendacao_calculada, # Usa nossa recomendação calculada
        'setor': setor_acao,
        'metodo_valuation': metodo, # Mantém o método intencionado
        'ajuste_analista': ajuste # Renomeado de ajuste_tecnico
    }

# Função para converter DataFrame para formato serializável (mantida)
def df_to_serializable(df):
    if df is None or df.empty:
        return {}
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime('%Y-%m-%d')
    return df.to_dict(orient='index')

# Função para converter dicionário serializado de volta para DataFrame (mantida)
def serializable_to_df(serial_dict):
    if not serial_dict:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(serial_dict, orient='index')
    try:
        df.index = pd.to_datetime(df.index)
    except ValueError:
        st.warning("Não foi possível converter o índice do DataFrame histórico para data.")
    return df

# Função principal para buscar e processar todos os dados (agora usa yfinance)
# @st.cache_data(ttl=3600) # Cache de 1 hora - REMOVIDO TEMPORARIAMENTE PARA TESTE
def atualizar_dados_completos_yfinance(acoes):
    st.info(f"Iniciando atualização para {len(acoes)} ações via yfinance...")
    resultados = {}
    progress_bar = st.progress(0)
    total_acoes = len(acoes)
    erros_busca = 0

    for i, acao in enumerate(acoes):
        # st.write(f"Processando {acao}...") # Debug
        hist, info, recom = buscar_dados_yfinance(acao)

        if hist.empty or not info:
            st.warning(f"Falha ao buscar dados completos para {acao}. Pulando.")
            erros_busca += 1
        else:
            dados_processados = processar_dados_acao(acao, hist, info, recom)
            if dados_processados:
                resultados[acao] = dados_processados
            else:
                erros_busca += 1

        progress_bar.progress((i + 1) / total_acoes)
        time.sleep(0.2) # Pausa menor, yfinance geralmente lida com rate limit

    if erros_busca > 0:
        st.warning(f"Coleta concluída, mas falhou para {erros_busca} de {total_acoes} ações.")
    else:
        st.success("Coleta de dados via yfinance concluída!")

    # Criar DataFrame resumido
    resumo_list = []
    for simbolo, dados_acao in resultados.items():
        val = dados_acao.get('valuation', {})
        info_res = dados_acao.get('info', {})
        preco_atual_val = val.get('preco_atual')
        preco_justo_val = val.get('preco_justo')

        potencial_val = np.nan
        if pd.notna(preco_atual_val) and pd.notna(preco_justo_val) and preco_atual_val > 0:
            potencial_val = ((preco_justo_val / preco_atual_val - 1) * 100)

        resumo_list.append({
            'Símbolo': simbolo.replace('.SA', ''),
            'Nome': info_res.get('name', 'N/A'),
            'Preço Atual': preco_atual_val,
            'Preço Justo': preco_justo_val,
            'Preço Compra Forte': val.get('preco_compra_forte'),
            'Potencial': potencial_val,
            'Recomendação': val.get('recomendacao', 'Indefinido'),
            'Setor': val.get('setor', 'N/A'), # Usa o setor calculado/mapeado
            'Método': val.get('metodo_valuation', 'N/A')
        })

    df_resumo = pd.DataFrame(resumo_list)
    df_resumo = df_resumo.fillna({'Potencial': 0})

    st.success("Processamento e resumo concluídos!")
    return resultados, df_resumo

# --- Funções de Exibição (Ajustadas para novos dados/insights) ---

# Função para converter dados históricos (usa serializable_to_df)
def converter_historico_para_df(historico_serializado):
    return serializable_to_df(historico_serializado)

# Função para criar gráfico de preços (sem grandes mudanças)
def criar_grafico_precos(df_historico, ticker, preco_justo, preco_compra_forte):
    # ... (código anterior mantido, com verificações pd.notna) ...
    if df_historico.empty or 'close' not in df_historico.columns or df_historico['close'].isnull().all():
        st.warning(f"Dados históricos de preço insuficientes ou inválidos para {ticker}.")
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_historico.index,
        y=df_historico['close'],
        mode='lines',
        name='Preço',
        line=dict(color='royalblue', width=2)
    ))

    if not df_historico.empty:
        ultimo_indice = df_historico.index[-1]
        primeiro_indice = df_historico.index[0]

        if pd.notna(preco_justo):
            fig.add_trace(go.Scatter(
                x=[primeiro_indice, ultimo_indice],
                y=[preco_justo, preco_justo],
                mode='lines',
                name='Preço Justo (Calc)',
                line=dict(color='green', width=1, dash='dash')
            ))

        if pd.notna(preco_compra_forte):
            fig.add_trace(go.Scatter(
                x=[primeiro_indice, ultimo_indice],
                y=[preco_compra_forte, preco_compra_forte],
                mode='lines',
                name='Compra Forte (Calc)',
                line=dict(color='darkgreen', width=1, dash='dot')
            ))

    fig.update_layout(
        title=f'Histórico de Preços - {ticker}',
        xaxis_title='Data',
        yaxis_title='Preço (R$)',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    return fig

# Função para criar gráfico de radar (sem grandes mudanças)
def criar_grafico_radar(df_resumo, ticker, setor):
    # ... (código anterior mantido, com tratamento de NaNs) ...
    if df_resumo is None or df_resumo.empty or setor is None:
        return None

    df_setor = df_resumo[df_resumo['Setor'] == setor].copy()
    if len(df_setor) <= 1: return None

    metricas = ['Preço Atual', 'Preço Justo', 'Potencial']
    for metrica in metricas:
        if metrica not in df_setor.columns: return None

    df_norm = df_setor.copy()
    for metrica in metricas:
        df_norm[metrica] = pd.to_numeric(df_norm[metrica], errors='coerce')
    df_norm.dropna(subset=metricas, inplace=True)
    df_norm = df_norm[np.isfinite(df_norm[metricas]).all(axis=1)]

    if df_norm.empty or len(df_norm) <= 1: return None

    metricas_norm_cols = []
    for metrica in metricas:
        col_norm = f'{metrica}_norm'
        metricas_norm_cols.append(col_norm)
        max_val = df_norm[metrica].max()
        min_val = df_norm[metrica].min()
        if max_val != min_val and pd.notna(max_val) and pd.notna(min_val):
            df_norm[col_norm] = (df_norm[metrica] - min_val) / (max_val - min_val)
        else:
            df_norm[col_norm] = 0.5

    fig = go.Figure()
    for _, acao in df_norm.iterrows():
        valores = acao[metricas_norm_cols].tolist()
        simbolo_acao = acao['Símbolo']
        largura = 3 if simbolo_acao == ticker else 1
        opacidade = 1.0 if simbolo_acao == ticker else 0.7
        fig.add_trace(go.Scatterpolar(
            r=valores, theta=metricas, fill='toself', name=simbolo_acao,
            line=dict(width=largura), opacity=opacidade
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f'Comparação Setorial - {setor}', showlegend=True, height=400
    )
    return fig

# Função para criar mapa de calor (sem grandes mudanças)
def criar_mapa_calor(df_resumo):
    # ... (código anterior mantido, com tratamento de NaNs) ...
    if df_resumo is None or df_resumo.empty or 'Recomendação' not in df_resumo.columns or 'Setor' not in df_resumo.columns:
        st.warning("Dados insuficientes no resumo para gerar o mapa de calor.")
        return go.Figure(), pd.DataFrame()

    mapa_recomendacao = {'Compra Forte': 5, 'Compra': 4, 'Neutro': 3, 'Venda Parcial': 2, 'Venda': 1}
    df_mapa = df_resumo.copy()
    if 'Recomendação' in df_mapa.columns:
        df_mapa['Valor_Recomendacao'] = df_mapa['Recomendação'].map(mapa_recomendacao).fillna(0)
    else: return go.Figure(), pd.DataFrame()

    if 'Setor' in df_mapa.columns:
        df_setor = df_mapa.groupby('Setor')['Valor_Recomendacao'].mean().reset_index()
        df_setor.dropna(subset=['Valor_Recomendacao'], inplace=True)
    else: return go.Figure(), pd.DataFrame()

    if df_setor.empty: return go.Figure(), df_setor

    def get_recomendacao_media(x):
        if pd.isna(x): return 'Indefinido'
        if x >= 4.5: return 'Compra Forte'
        if x >= 3.5: return 'Compra'
        if x >= 2.5: return 'Neutro'
        if x >= 1.5: return 'Venda Parcial'
        if x > 0: return 'Venda'
        return 'Indefinido'
    df_setor['Recomendação_Média'] = df_setor['Valor_Recomendacao'].apply(get_recomendacao_media)

    try:
        df_setor_sorted = df_setor.sort_values('Valor_Recomendacao', ascending=False)
        pivot_table = df_setor_sorted.set_index('Setor')[['Valor_Recomendacao']]
        if pivot_table.empty: return go.Figure(), df_setor

        fig = px.imshow(
            pivot_table.values, labels=dict(x="", y="Setor", color="Score Médio"),
            y=pivot_table.index, x=['Recomendação Média'],
            color_continuous_scale=['red', 'yellow', 'green'], range_color=[1, 5],
            text_auto='.2f', aspect="auto"
        )
        fig.update_xaxes(showticklabels=False)
        fig.update_layout(
            title='Recomendação Média por Setor (Calculada)',
            height=max(300, len(pivot_table.index) * 25 + 50),
            coloraxis_colorbar=dict(title="Score")
        )
    except Exception as e:
        st.error(f"Erro ao criar o mapa de calor: {e}")
        return go.Figure(), df_setor

    return fig, df_setor_sorted

# Função para exibir detalhes da ação (adaptada para mostrar insights yfinance)
def exibir_detalhes_acao(dados_completos, ticker_sa, df_resumo):
    ticker = ticker_sa.replace('.SA', '')
    if dados_completos is None or ticker_sa not in dados_completos:
        st.error(f"Dados não disponíveis para {ticker_sa}. Tente atualizar os dados.")
        return

    acao_data = dados_completos[ticker_sa]
    info = acao_data.get('info', {})
    valuation = acao_data.get('valuation', {})
    insights = acao_data.get('insights', {})
    df_historico = converter_historico_para_df(acao_data.get('historico', {}))

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"{info.get('name', 'Nome Indisponível')} ({ticker})")
        st.write(f"**Setor (yfinance):** {info.get('sector', 'N/A')}")
        st.write(f"**Indústria (yfinance):** {info.get('industry', 'N/A')}")
        st.write(f"**Método Valuation (Intenção):** {valuation.get('metodo_valuation', 'N/A')}")

        fig_precos = criar_grafico_precos(
            df_historico, ticker,
            valuation.get('preco_justo'), valuation.get('preco_compra_forte')
        )
        st.plotly_chart(fig_precos, use_container_width=True)

    with col2:
        # Card de recomendação (baseado na nossa valuation)
        recomendacao_calc = valuation.get('recomendacao', 'Indefinido')
        cor_recomendacao = {'Compra Forte': 'darkgreen', 'Compra': 'green', 'Neutro': 'gray', 'Venda Parcial': 'orange', 'Venda': 'red'}.get(recomendacao_calc, 'black')
        preco_atual = valuation.get('preco_atual', np.nan)
        preco_justo = valuation.get('preco_justo', np.nan)
        preco_compra_forte = valuation.get('preco_compra_forte', np.nan)

        potencial_str = "N/A"
        if pd.notna(preco_atual) and pd.notna(preco_justo) and preco_atual > 0:
            potencial = ((preco_justo / preco_atual - 1) * 100)
            potencial_str = f"{potencial:.2f}%"

        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
            <h4 style="text-align: center; margin-bottom: 5px;">Recomendação (Calculada)</h4>
            <h3 style="text-align: center; color: {cor_recomendacao}; margin-top: 0px; margin-bottom: 10px;">{recomendacao_calc}</h3>
            <hr style="margin-top: 5px; margin-bottom: 10px;">
            <p style="font-size: 0.9em; margin-bottom: 3px;"><strong>Preço Atual:</strong> R$ {preco_atual:.2f if pd.notna(preco_atual) else 'N/A'}</p>
            <p style="font-size: 0.9em; margin-bottom: 3px;"><strong>Preço Justo (Calc):</strong> R$ {preco_justo:.2f if pd.notna(preco_justo) else 'N/A'}</p>
            <p style="font-size: 0.9em; margin-bottom: 3px;"><strong>Compra Forte (Calc):</strong> R$ {preco_compra_forte:.2f if pd.notna(preco_compra_forte) else 'N/A'}</p>
            <p style="font-size: 0.9em; margin-bottom: 0px;"><strong>Potencial (Calc):</strong> {potencial_str}</p>
        </div>
        """, unsafe_allow_html=True)

        # Mostrar Insights Simplificados (Recomendação Analistas)
        st.subheader("Insights de Analistas (yfinance)")
        rating_analista = insights.get('recommendation_rating', 'N/A')
        target_analista = insights.get('target_price_mean', np.nan)
        count_analista = insights.get('analyst_count', 0)

        st.metric(label="Rating Médio Analistas", value=rating_analista)
        st.metric(label="Preço Alvo Médio Analistas", value=f"R$ {target_analista:.2f}" if pd.notna(target_analista) else "N/A")
        st.metric(label="Número de Analistas", value=count_analista if count_analista else "N/A")

        # Outros dados do info
        st.subheader("Outros Dados (yfinance)")
        st.metric(label="Market Cap", value=f"R$ {info.get('marketCap', 0) / 1e9:.2f} Bi" if info.get('marketCap') else "N/A")
        st.metric(label="Beta", value=f"{info.get('beta'):.2f}" if info.get('beta') else "N/A")
        st.metric(label="P/L (TTM)", value=f"{info.get('trailingPE'):.2f}" if info.get('trailingPE') else "N/A")
        st.metric(label="Dividend Yield", value=f"{info.get('dividendYield', 0) * 100:.2f}%" if info.get('dividendYield') else "N/A")

    # Seção de análise comparativa (usa setor calculado)
    st.subheader("Análise Comparativa Setorial")
    setor_acao_calc = valuation.get('setor') # Usa o setor que usamos para calcular
    if setor_acao_calc and df_resumo is not None and not df_resumo.empty:
        col1_comp, col2_comp = st.columns([1, 2])
        with col1_comp:
            radar_fig = criar_grafico_radar(df_resumo, ticker, setor_acao_calc)
            if radar_fig: st.plotly_chart(radar_fig, use_container_width=True)
            else: st.info(f"Dados insuficientes no setor '{setor_acao_calc}' para radar.")
        with col2_comp:
            df_setor_comp = df_resumo[df_resumo['Setor'] == setor_acao_calc].copy()
            if not df_setor_comp.empty:
                def highlight_row(row):
                    return ['background-color: #e6f2ff'] * len(row) if row['Símbolo'] == ticker else [''] * len(row)
                cols_to_show = ['Símbolo', 'Nome', 'Preço Atual', 'Preço Justo', 'Potencial', 'Recomendação']
                cols_present = [col for col in cols_to_show if col in df_setor_comp.columns]
                df_display_comp = df_setor_comp[cols_present].copy()
                # Formatação segura
                for col, fmt in [('Potencial', '{:.2f}%'), ('Preço Atual', 'R$ {:.2f}'), ('Preço Justo', 'R$ {:.2f}')]:
                    if col in df_display_comp.columns:
                        df_display_comp[col] = df_display_comp[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
                st.dataframe(
                    df_display_comp.style.apply(highlight_row, axis=1),
                    hide_index=True, use_container_width=True,
                    height=min(400, len(df_display_comp)*35 + 40)
                )
            else: st.info(f"Nenhuma outra ação encontrada no setor '{setor_acao_calc}' para comparação.")
    else:
        st.warning("Setor da ação não definido ou resumo indisponível para comparação.")

# Função para exibir visão geral (sem grandes mudanças)
def exibir_visao_geral(df_resumo):
    # ... (código anterior mantido, com tratamento de NaNs) ...
    if df_resumo is None or df_resumo.empty:
        st.warning("Resumo de dados indisponível. Tente atualizar os dados.")
        return

    st.subheader("Visão Geral do Mercado")
    col1_vg, col2_vg = st.columns([2, 1])
    with col1_vg:
        mapa_calor, df_setor_mapa = criar_mapa_calor(df_resumo)
        if mapa_calor: st.plotly_chart(mapa_calor, use_container_width=True)
        else: st.warning("Não foi possível gerar o mapa de calor.")
    with col2_vg:
        if not df_setor_mapa.empty:
            st.subheader("Recomendação Média")
            def highlight_recomendacao(val):
                colors = {'Compra Forte': 'background-color: #006100; color: white', 'Compra': 'background-color: #c6efce; color: #006100', 'Neutro': 'background-color: #ffeb9c; color: #9c6500', 'Venda Parcial': 'background-color: #ffcc99; color: #9c3400', 'Venda': 'background-color: #ffc7ce; color: #9c0006'}
                return colors.get(val, '')
            st.dataframe(
                df_setor_mapa[['Setor', 'Recomendação_Média']].style.applymap(highlight_recomendacao, subset=['Recomendação_Média']),
                hide_index=True, use_container_width=True,
                height=min(400, len(df_setor_mapa)*35 + 40)
            )
        else: st.info("Dados de recomendação por setor indisponíveis.")

    st.divider()
    st.subheader("Distribuição e Destaques")
    col1_dist, col2_dist = st.columns([1, 2])
    with col1_dist:
        st.markdown("**Distribuição (Calculada)**")
        if 'Recomendação' in df_resumo.columns:
            recomendacoes_count = df_resumo['Recomendação'].value_counts().reset_index()
            recomendacoes_count.columns = ['Recomendação', 'Contagem']
            ordem = ['Compra Forte', 'Compra', 'Neutro', 'Venda Parcial', 'Venda', 'Indefinido']
            recomendacoes_count['ordem'] = pd.Categorical(recomendacoes_count['Recomendação'], categories=ordem, ordered=True)
            recomendacoes_count = recomendacoes_count.sort_values('ordem').drop('ordem', axis=1)
            cores = {'Compra Forte': 'darkgreen', 'Compra': 'green', 'Neutro': 'gray', 'Venda Parcial': 'orange', 'Venda': 'red', 'Indefinido': 'black'}
            fig_dist = px.bar(recomendacoes_count, x='Recomendação', y='Contagem', color='Recomendação', color_discrete_map=cores, text='Contagem')
            fig_dist.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_dist, use_container_width=True)
        else: st.info("Dados de recomendação indisponíveis.")
    with col2_dist:
        st.markdown("**Top 5 Potencial (Calculado)**")
        if 'Potencial' in df_resumo.columns:
            df_resumo_pot = df_resumo.copy()
            df_resumo_pot['Potencial_Num'] = pd.to_numeric(df_resumo_pot['Potencial'], errors='coerce')
            top_potencial = df_resumo_pot.sort_values('Potencial_Num', ascending=False, na_position='last').head(5)
            cols_top = ['Símbolo', 'Nome', 'Potencial', 'Preço Atual', 'Preço Justo', 'Recomendação']
            cols_present_top = [col for col in cols_top if col in top_potencial.columns]
            df_display_top = top_potencial[cols_present_top].copy()
            # Formatação segura
            for col, fmt in [('Potencial', '{:.2f}%'), ('Preço Atual', 'R$ {:.2f}'), ('Preço Justo', 'R$ {:.2f}')]:
                 if col in df_display_top.columns:
                     df_display_top[col] = df_display_top[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
            st.dataframe(df_display_top, hide_index=True, use_container_width=True)
        else: st.info("Dados de potencial indisponíveis.")

    st.divider()
    st.subheader("Tabela Completa de Ações")
    def highlight_recomendacao_row(row):
        rec = row.get('Recomendação', '')
        if rec == 'Compra Forte': return ['background-color: #c6efce'] * len(row)
        if rec == 'Compra': return ['background-color: #e6f2ff'] * len(row)
        if rec == 'Venda': return ['background-color: #ffc7ce'] * len(row)
        return [''] * len(row)
    df_display_all = df_resumo.copy()
    # Formatação segura
    for col, fmt in [('Potencial', '{:.2f}%'), ('Preço Atual', 'R$ {:.2f}'), ('Preço Justo', 'R$ {:.2f}'), ('Preço Compra Forte', 'R$ {:.2f}')]:
        if col in df_display_all.columns:
            df_display_all[col] = df_display_all[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
    cols_all = ['Símbolo', 'Nome', 'Setor', 'Preço Atual', 'Preço Justo', 'Potencial', 'Recomendação', 'Método']
    cols_present_all = [col for col in cols_all if col in df_display_all.columns]
    df_display_final = df_display_all[cols_present_all]
    st.dataframe(df_display_final.style.apply(highlight_recomendacao_row, axis=1), hide_index=True, use_container_width=True)

# Função para exibir metodologia (atualizada para refletir yfinance e simplificações)
def exibir_metodologia():
    st.subheader("Metodologia de Análise e Recomendação (v. yfinance)")
    st.markdown("""
    ### Fontes de Dados
    Os dados históricos, informações da empresa e recomendações de analistas são coletados diretamente da biblioteca `yfinance` no momento da atualização.

    ### Cálculo de Valuation (Simplificado)
    A determinação do "Preço Justo" e "Preço Compra Forte" é feita através de um modelo *heurístico* que considera:
    1.  **Setor da Ação:** Ações são agrupadas em setores (Financeiro, Utilities, etc.) com base nos dados do yfinance ou em uma lista pré-definida.
    2.  **Parâmetros Setoriais:** Cada setor possui parâmetros *exemplo* (como P/L justo, Dividend Yield justo) definidos no código.
    3.  **Dados da Ação (yfinance):** Utiliza o Preço Atual, P/L (quando disponível), Dividend Yield (quando disponível) e Beta (quando disponível) obtidos via `yfinance`.
    4.  **Cálculo:**
        *   Tenta-se aplicar uma lógica baseada nos múltiplos ou yields. Ex: Se o P/L atual é X e o P/L justo do setor é Y, o Preço Justo seria `Preço Atual * (Y / X)`.
        *   Se os dados específicos (P/L, DY) não estão disponíveis, usa-se uma heurística mais genérica comparando o P/L justo do setor com um P/L médio de mercado (proxy = 12).
        *   Um pequeno ajuste (+/- 5%) pode ser aplicado com base na recomendação média dos analistas (Compra/Venda) obtida do `yfinance`.
        *   O resultado é limitado para evitar valores extremos (+/- 50% do preço atual).
    5.  **Preço Compra Forte:** Definido como 80% do Preço Justo calculado.

    **Importante:** Este cálculo é uma *estimativa simplificada* e não substitui uma análise fundamentalista profunda, que exigiria mais dados (fluxo de caixa, balanços, etc.).

    ### Modelo de Recomendação (Calculada)
    A recomendação exibida ("Compra Forte", "Compra", "Neutro", etc.) é *calculada* com base na comparação entre o Preço Atual e os níveis de Preço Justo e Preço Compra Forte estimados pelo modelo descrito acima.

    *Disclaimer: Esta análise é gerada por um modelo automatizado com simplificações e não constitui recomendação de investimento personalizada. Consulte um profissional financeiro antes de tomar decisões.*
    """)

# --- Interface Principal --- 
st.title("📈 Análise de Ações Brasileiras - Modelo SPX FALCON (v. yfinance)")

# --- Gerenciamento de Estado e Botão de Atualização --- 
if 'dados_acoes' not in st.session_state: st.session_state['dados_acoes'] = None
if 'resumo_acoes' not in st.session_state: st.session_state['resumo_acoes'] = None
if 'ultima_atualizacao' not in st.session_state: st.session_state['ultima_atualizacao'] = None

st.sidebar.title("Controles")

if st.sidebar.button("Buscar/Atualizar Dados (yfinance)", key="update_button_yf"):
    with st.spinner("Atualizando dados via yfinance... Isso pode levar alguns minutos."):
        try:
            # Limpa o cache da função específica se existir (opcional, mas garante)
            # buscar_dados_yfinance.clear()
            # Chama a função de atualização que usa yfinance
            dados_atualizados, resumo_atualizado = atualizar_dados_completos_yfinance(acoes_default)
            st.session_state['dados_acoes'] = dados_atualizados
            st.session_state['resumo_acoes'] = resumo_atualizado
            st.session_state['ultima_atualizacao'] = datetime.now()
            st.sidebar.success("Dados atualizados com sucesso!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Erro durante a atualização yfinance: {e}")

# Carrega dados do estado da sessão
dados_carregados = st.session_state['dados_acoes']
resumo_carregado = st.session_state['resumo_acoes']

# Tenta carregar na primeira vez se não houver dados
if dados_carregados is None:
    st.info("Dados não carregados. Tentando buscar dados iniciais via yfinance...")
    with st.spinner("Buscando dados iniciais..."):
        try:
            dados_carregados, resumo_carregado = atualizar_dados_completos_yfinance(acoes_default)
            st.session_state['dados_acoes'] = dados_carregados
            st.session_state['resumo_acoes'] = resumo_carregado
            st.session_state['ultima_atualizacao'] = datetime.now()
            st.rerun()
        except Exception as e:
            st.error(f"Falha ao buscar dados iniciais: {e}")

# Exibe data da última atualização
if st.session_state['ultima_atualizacao']:
    st.sidebar.caption(f"Última atualização: {st.session_state['ultima_atualizacao'].strftime('%d/%m/%Y %H:%M:%S')}")
else:
    st.sidebar.caption("Dados ainda não atualizados.")

# --- Exibição Principal --- 
if dados_carregados is not None and resumo_carregado is not None:
    st.sidebar.header("Visualização")
    lista_acoes_disponiveis = sorted(list(dados_carregados.keys()))
    nomes_formatados = []
    mapa_nome_ticker = {}
    if lista_acoes_disponiveis:
        for ticker_sa in lista_acoes_disponiveis:
            nome = dados_carregados[ticker_sa].get('info', {}).get('name', ticker_sa.replace('.SA', ''))
            nome_fmt = f"{nome} ({ticker_sa.replace('.SA', '')})"
            nomes_formatados.append(nome_fmt)
            mapa_nome_ticker[nome_fmt] = ticker_sa
    else:
        st.warning("Nenhuma ação foi carregada com sucesso.")

    opcoes_view = ["Visão Geral do Mercado", "Detalhes por Ação", "Metodologia"]
    view_selecionada = st.sidebar.selectbox("Escolha a visualização:", opcoes_view, key="view_select_yf")

    if view_selecionada == "Visão Geral do Mercado":
        exibir_visao_geral(resumo_carregado)
    elif view_selecionada == "Detalhes por Ação":
        if nomes_formatados:
            nome_acao_selecionada = st.selectbox("Selecione uma Ação:", options=nomes_formatados, key="stock_select_yf")
            ticker_selecionado_sa = mapa_nome_ticker.get(nome_acao_selecionada)
            if ticker_selecionado_sa:
                exibir_detalhes_acao(dados_carregados, ticker_selecionado_sa, resumo_carregado)
            else: st.error("Erro ao encontrar o ticker.")
        else: st.warning("Nenhuma ação disponível para seleção.")
    elif view_selecionada == "Metodologia":
        exibir_metodologia()
else:
    st.error("Não foi possível carregar ou atualizar os dados via yfinance. Verifique a conexão ou tente atualizar novamente.")
