# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import time
import sys

# Tenta importar ApiClient do ambiente sandbox
try:
    sys.path.append("/opt/.manus/.sandbox-runtime")
    from data_api import ApiClient
    # Inicializa o cliente da API globalmente ou onde for necessário
    # É importante garantir que ele seja inicializado apenas uma vez se possível
    # ou dentro da função de atualização para evitar problemas de estado.
    # Para este exemplo, inicializaremos dentro da função de atualização.
except ImportError:
    st.error("Erro crítico: Não foi possível importar ApiClient. Verifique o ambiente.")
    # Define um cliente dummy para evitar erros posteriores, mas a funcionalidade de API não funcionará
    class ApiClient:
        def call_api(self, *args, **kwargs):
            st.warning("ApiClient não está disponível. A busca de dados não funcionará.")
            return None
    # client = ApiClient() # Não inicializar aqui se for feito na função

# Configuração da página
st.set_page_config(
    page_title="Análise de Ações Brasileiras - Modelo SPX FALCON",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Lógica de Coleta e Processamento de Dados (Integrada) ---

# Lista de ações (pode ser configurável no futuro)
acoes_default = [
    'AGRO3.SA', 'BBAS3.SA', 'BBSE3.SA', 'BPAC11.SA', 'EGIE3.SA', 'ITUB3.SA',
    'PRIO3.SA', 'PSSA3.SA', 'SAPR3.SA', 'SBSP3.SA', 'VIVT3.SA', 'WEGE3.SA',
    'TOTS3.SA', 'B3SA3.SA', 'TAEE3.SA', 'CMIG3.SA',
    'PETR4.SA', 'VALE3.SA', 'ITSA4.SA', 'ABEV3.SA', 'RENT3.SA', 'MGLU3.SA',
    'SUZB3.SA', 'RADL3.SA', 'CSAN3.SA', 'EQTL3.SA'
]

# Definir parâmetros setoriais para valuation
setores = {
    'Financeiro': {
        'acoes': ['BBAS3.SA', 'ITUB3.SA', 'B3SA3.SA', 'BPAC11.SA', 'BBSE3.SA', 'PSSA3.SA', 'ITSA4.SA'], # Adicionado ITSA4
        'metodo': 'Gordon Growth',
        'parametros': {
            'crescimento_longo_prazo': 0.04,
            'premio_risco': 0.06,
            'beta_medio': 0.9,
            'p_vp_justo': 1.8
        }
    },
    'Utilities': {
        'acoes': ['EGIE3.SA', 'CMIG3.SA', 'SBSP3.SA', 'SAPR3.SA', 'TAEE3.SA', 'EQTL3.SA'],
        'metodo': 'Fluxo de Caixa Descontado',
        'parametros': {
            'crescimento_longo_prazo': 0.03,
            'premio_risco': 0.05,
            'beta_medio': 0.7,
            'dividend_yield_justo': 0.055
        }
    },
    'Tecnologia': {
        'acoes': ['TOTS3.SA', 'WEGE3.SA'],
        'metodo': 'Múltiplos de Receita',
        'parametros': {
            'crescimento_longo_prazo': 0.08,
            'premio_risco': 0.07,
            'beta_medio': 1.1,
            'ps_ratio_justo': 3.5
        }
    },
    'Commodities': {
        'acoes': ['AGRO3.SA', 'PRIO3.SA', 'VALE3.SA', 'SUZB3.SA', 'CSAN3.SA', 'PETR4.SA'], # Adicionado PETR4
        'metodo': 'Ciclo de Preços',
        'parametros': {
            'crescimento_longo_prazo': 0.03,
            'premio_risco': 0.08,
            'beta_medio': 1.2,
            'ev_ebitda_justo': 5.5
        }
    },
    'Consumo': {
        'acoes': ['VIVT3.SA', 'ABEV3.SA', 'RENT3.SA', 'MGLU3.SA', 'RADL3.SA'],
        'metodo': 'Múltiplos Comparáveis',
        'parametros': {
            'crescimento_longo_prazo': 0.05,
            'premio_risco': 0.06,
            'beta_medio': 0.85,
            'p_l_justo': 16
        }
    }
}

# Função para coletar dados históricos de preços (adaptada de collect_stock_data.py)
def coletar_dados_historicos(client, simbolo):
    # st.write(f"Coletando dados históricos para {simbolo}...") # Debug
    try:
        dados = client.call_api('YahooFinance/get_stock_chart', query={
            'symbol': simbolo,
            'interval': '1mo',
            'range': '5y',
            'region': 'BR',
            'includeAdjustedClose': True
        })
        
        if not dados or 'chart' not in dados or 'result' not in dados['chart'] or not dados['chart']['result']:
            st.warning(f"Sem dados históricos para {simbolo}")
            return None
        
        result = dados['chart']['result'][0]
        timestamps = result.get('timestamp', [])
        quotes_list = result.get('indicators', {}).get('quote', [])
        
        if not timestamps or not quotes_list:
             st.warning(f"Estrutura de dados históricos inesperada para {simbolo}")
             return None
             
        quotes = quotes_list[0]
        if len(timestamps) != len(quotes.get('close', [])):
             st.warning(f"Inconsistência no tamanho dos dados históricos para {simbolo}")
             # Tenta usar o menor tamanho comum
             min_len = min(len(timestamps), len(quotes.get('close', [])), len(quotes.get('open', [])), 
                           len(quotes.get('high', [])), len(quotes.get('low', [])), len(quotes.get('volume', [])))
             timestamps = timestamps[:min_len]
             for key in quotes:
                 if isinstance(quotes[key], list):
                     quotes[key] = quotes[key][:min_len]

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': quotes.get('open'),
            'high': quotes.get('high'),
            'low': quotes.get('low'),
            'close': quotes.get('close'),
            'volume': quotes.get('volume')
        })
        
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('date', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        df = df.dropna(subset=['close']) # Remove dias sem fechamento
        
        meta = result.get('meta', {})
        info = {
            'symbol': meta.get('symbol', simbolo),
            'name': meta.get('shortName', 'N/A'),
            'currency': meta.get('currency', 'BRL'),
            'exchange': meta.get('exchangeName', 'SAO'),
            'current_price': meta.get('regularMarketPrice'), # Pode ser None
            '52w_high': meta.get('fiftyTwoWeekHigh'),
            '52w_low': meta.get('fiftyTwoWeekLow')
        }
        
        # Se current_price for None, tenta pegar o último fechamento
        if info['current_price'] is None and not df.empty:
            info['current_price'] = df['close'].iloc[-1]
            
        if info['current_price'] is None:
             st.warning(f"Não foi possível obter o preço atual para {simbolo}")
             return None # Retorna None se não conseguir preço atual

        return {'data': df, 'info': info}
    
    except Exception as e:
        st.error(f"Erro ao coletar dados históricos para {simbolo}: {e}")
        return None

# Função para coletar insights e valuation (adaptada de collect_stock_data.py)
def coletar_insights(client, simbolo):
    # st.write(f"Coletando insights para {simbolo}...") # Debug
    try:
        insights = client.call_api('YahooFinance/get_stock_insights', query={'symbol': simbolo})
        
        if not insights or 'finance' not in insights or 'result' not in insights['finance']:
            # st.warning(f"Sem insights para {simbolo}") # Comum não ter insights
            return None
        
        result = insights['finance']['result']
        valuation_data = {}
        recommendation = {}
        technical_events = {}

        if result:
            if 'instrumentInfo' in result and result['instrumentInfo'] and 'valuation' in result['instrumentInfo']:
                valuation = result['instrumentInfo']['valuation']
                if valuation: # Verifica se valuation não é None
                    valuation_data = {
                        'description': valuation.get('description'),
                        'discount': valuation.get('discount'),
                        'relative_value': valuation.get('relativeValue')
                    }
            
            if 'recommendation' in result and result['recommendation']:
                rec = result['recommendation']
                recommendation = {
                    'target_price': rec.get('targetPrice'),
                    'rating': rec.get('rating')
                }
            
            if 'instrumentInfo' in result and result['instrumentInfo'] and 'technicalEvents' in result['instrumentInfo']:
                tech = result['instrumentInfo']['technicalEvents']
                if tech: # Verifica se tech não é None
                    if 'shortTermOutlook' in tech and tech['shortTermOutlook']:
                        technical_events['short_term'] = {
                            'direction': tech['shortTermOutlook'].get('direction'),
                            'score': tech['shortTermOutlook'].get('score'),
                            'description': tech['shortTermOutlook'].get('scoreDescription')
                        }
                    if 'intermediateTermOutlook' in tech and tech['intermediateTermOutlook']:
                        technical_events['mid_term'] = {
                            'direction': tech['intermediateTermOutlook'].get('direction'),
                            'score': tech['intermediateTermOutlook'].get('score'),
                            'description': tech['intermediateTermOutlook'].get('scoreDescription')
                        }
                    if 'longTermOutlook' in tech and tech['longTermOutlook']:
                        technical_events['long_term'] = {
                            'direction': tech['longTermOutlook'].get('direction'),
                            'score': tech['longTermOutlook'].get('score'),
                            'description': tech['longTermOutlook'].get('scoreDescription')
                        }
        
        return {
            'valuation': valuation_data,
            'recommendation': recommendation,
            'technical': technical_events
        }
    
    except Exception as e:
        st.error(f"Erro ao coletar insights para {simbolo}: {e}")
        return None

# Função para calcular preço justo baseado no setor (adaptada de collect_stock_data.py)
def calcular_preco_justo(simbolo, preco_atual, insights):
    setor_acao = None
    parametros = None
    metodo = None

    # Encontrar o setor da ação
    for nome_setor, info in setores.items():
        if simbolo in info['acoes']:
            setor_acao = nome_setor
            parametros = info['parametros']
            metodo = info['metodo']
            break
    
    # Se não encontrou setor específico, usa parâmetros genéricos (ou retorna erro/aviso)
    if not setor_acao:
        st.warning(f"Setor não definido para {simbolo}. Usando parâmetros genéricos.")
        setor_acao = 'Geral'
        parametros = {
            'crescimento_longo_prazo': 0.05, 'premio_risco': 0.06,
            'beta_medio': 1.0, 'p_l_justo': 14
        }
        metodo = 'Múltiplos Gerais'

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
        
    # Ajustar com base nos insights técnicos, se disponíveis
    ajuste = 1.0
    if insights and 'technical' in insights and insights['technical']:
        if 'long_term' in insights['technical'] and insights['technical']['long_term']:
            direction = insights['technical']['long_term'].get('direction')
            score = insights['technical']['long_term'].get('score')
            if direction == 'up' and isinstance(score, (int, float)):
                ajuste = 1.0 + (score / 10.0)
            elif direction == 'down' and isinstance(score, (int, float)):
                ajuste = 1.0 - (score / 10.0)
            ajuste = max(0.5, min(1.5, ajuste)) # Limita o ajuste

    # Calcular preço justo baseado no método do setor
    preco_justo = np.nan # Default
    try:
        if metodo == 'Gordon Growth' and 'p_vp_justo' in parametros:
            # Exemplo: P/VP. Pode precisar de dados adicionais (VPA) que não temos aqui.
            # Simplificação: Usando um multiplicador sobre o preço atual como proxy.
            # Idealmente, buscaria VPA e usaria P/VP = Preço / VPA.
            # Vamos usar uma heurística baseada no P/L médio vs P/L justo como alternativa
            preco_justo = preco_atual * (parametros.get('p_l_justo', 15) / 12.0) * ajuste 
        elif metodo == 'Fluxo de Caixa Descontado' and 'dividend_yield_justo' in parametros:
            # Exemplo: Dividend Yield. Pode precisar de Dividendo por Ação.
            # Simplificação: Invertendo a lógica do yield.
            # Se yield justo é 5.5%, e yield atual é X, Preço Justo = Preço Atual * (Yield Atual / Yield Justo)
            # Sem Yield Atual, usamos heurística de P/L.
            preco_justo = preco_atual * (parametros.get('p_l_justo', 15) / 12.0) * ajuste
        elif metodo == 'Múltiplos de Receita' and 'ps_ratio_justo' in parametros:
            # Exemplo: PS Ratio. Precisa de Receita por Ação.
            # Simplificação: Heurística de P/L.
            preco_justo = preco_atual * (parametros.get('p_l_justo', 15) / 12.0) * ajuste
        elif metodo == 'Ciclo de Preços' and 'ev_ebitda_justo' in parametros:
            # Exemplo: EV/EBITDA. Precisa de EBITDA por Ação e Dívida Líquida.
            # Simplificação: Heurística de P/L.
            preco_justo = preco_atual * (parametros.get('p_l_justo', 15) / 12.0) * ajuste
        elif 'p_l_justo' in parametros: # Consumo ou Geral
            # Usando P/L justo como base principal
            # Idealmente, buscaria LPA (Lucro por Ação) e calcularia Preço Justo = LPA * P/L Justo
            # Simplificação: Ajusta preço atual por um fator P/L_Justo / P/L_Medio_Mercado (ex: 12)
            preco_justo = preco_atual * (parametros['p_l_justo'] / 12.0) * ajuste
        else:
             preco_justo = preco_atual * ajuste # Se nenhum método se aplica, usa só ajuste técnico
             
    except Exception as e:
        st.warning(f"Erro no cálculo do preço justo para {simbolo} ({metodo}): {e}")
        preco_justo = np.nan

    # Calcular preço de compra forte (20% abaixo do preço justo)
    preco_compra_forte = preco_justo * 0.8 if pd.notna(preco_justo) else np.nan
    
    # Determinar recomendação
    recomendacao = 'Indefinido'
    if pd.notna(preco_justo) and preco_atual > 0:
        if preco_atual < preco_compra_forte:
            recomendacao = "Compra Forte"
        elif preco_atual < preco_justo * 0.95:
            recomendacao = "Compra"
        elif preco_atual > preco_justo * 1.15:
            recomendacao = "Venda"
        elif preco_atual > preco_justo * 1.05:
            recomendacao = "Venda Parcial"
        else:
            recomendacao = "Neutro"
    
    return {
        'preco_atual': preco_atual,
        'preco_justo': preco_justo,
        'preco_compra_forte': preco_compra_forte,
        'recomendacao': recomendacao,
        'setor': setor_acao,
        'metodo_valuation': metodo,
        'ajuste_tecnico': ajuste
    }

# Função para converter DataFrame para formato serializável em JSON (necessário para cache/session_state)
def df_to_serializable(df):
    if df is None or df.empty:
        return {}
    # Converte o índice para string se for DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime('%Y-%m-%d')
    # Converte para dicionário orient='index' que é serializável
    return df.to_dict(orient='index')

# Função para converter dicionário serializado de volta para DataFrame
def serializable_to_df(serial_dict):
    if not serial_dict:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(serial_dict, orient='index')
    # Tenta converter o índice de volta para datetime
    try:
        df.index = pd.to_datetime(df.index)
    except ValueError:
        st.warning("Não foi possível converter o índice do DataFrame histórico para data.")
        # Mantém o índice como string se a conversão falhar
    return df

# Função principal para buscar e processar todos os dados
@st.cache_data(ttl=3600) # Cache por 1 hora
def atualizar_dados_completos(acoes):
    st.info(f"Iniciando atualização para {len(acoes)} ações...")
    client = ApiClient() # Inicializa o cliente aqui para ser thread-safe com cache
    resultados = {}
    progress_bar = st.progress(0)
    total_acoes = len(acoes)

    for i, acao in enumerate(acoes):
        # st.write(f"Processando {acao}...") # Debug
        dados_historicos = coletar_dados_historicos(client, acao)
        if not dados_historicos or dados_historicos['info']['current_price'] is None:
            st.warning(f"Dados históricos ou preço atual indisponíveis para {acao}. Pulando.")
            progress_bar.progress((i + 1) / total_acoes)
            continue
        
        insights = coletar_insights(client, acao)
        preco_atual = dados_historicos['info']['current_price']
        valuation = calcular_preco_justo(acao, preco_atual, insights)
        
        # Armazena resultados (DataFrame histórico é convertido para dict serializável)
        resultados[acao] = {
            'historico': df_to_serializable(dados_historicos['data']),
            'info': dados_historicos['info'],
            'insights': insights if insights else {},
            'valuation': valuation
        }
        
        progress_bar.progress((i + 1) / total_acoes)
        time.sleep(0.5) # Pequena pausa para não sobrecarregar a API

    st.success("Coleta de dados concluída!")
    
    # Criar DataFrame resumido
    resumo_list = []
    for simbolo, dados_acao in resultados.items():
        # Verifica se valuation existe e tem as chaves necessárias
        val = dados_acao.get('valuation', {})
        info = dados_acao.get('info', {})
        preco_atual_val = val.get('preco_atual')
        preco_justo_val = val.get('preco_justo')
        
        potencial_val = np.nan
        if pd.notna(preco_atual_val) and pd.notna(preco_justo_val) and preco_atual_val > 0:
            potencial_val = ((preco_justo_val / preco_atual_val - 1) * 100)
            
        resumo_list.append({
            'Símbolo': simbolo.replace('.SA', ''),
            'Nome': info.get('name', 'N/A'),
            'Preço Atual': preco_atual_val,
            'Preço Justo': preco_justo_val, # Mantém como float
            'Preço Compra Forte': val.get('preco_compra_forte'), # Mantém como float
            'Potencial': potencial_val, # Mantém como float
            'Recomendação': val.get('recomendacao', 'Indefinido'),
            'Setor': val.get('setor', 'N/A'),
            'Método': val.get('metodo_valuation', 'N/A')
        })

    df_resumo = pd.DataFrame(resumo_list)
    # Tratar NaNs que podem ter surgido
    df_resumo = df_resumo.fillna({'Potencial': 0}) # Ou outra estratégia
    
    st.success("Processamento e resumo concluídos!")
    return resultados, df_resumo

# --- Funções de Exibição (Modificadas para usar dados em memória) ---

# Função para converter dados históricos (agora usa serializable_to_df)
def converter_historico_para_df(historico_serializado):
    return serializable_to_df(historico_serializado)

# Função para criar gráfico de preços (sem grandes mudanças, mas verifica NaNs)
def criar_grafico_precos(df_historico, ticker, preco_justo, preco_compra_forte):
    # ... (código anterior, mas adiciona verificações pd.notna) ...
    if df_historico.empty or 'close' not in df_historico.columns or df_historico['close'].isnull().all():
        st.warning(f"Dados históricos de preço insuficientes ou inválidos para {ticker}.")
        return go.Figure()
        
    fig = go.Figure()
    
    # Adicionar linha de preço
    fig.add_trace(go.Scatter(
        x=df_historico.index,
        y=df_historico['close'],
        mode='lines',
        name='Preço',
        line=dict(color='royalblue', width=2)
    ))
    
    # Adicionar linhas de preço justo e compra forte
    if not df_historico.empty:
        ultimo_indice = df_historico.index[-1]
        primeiro_indice = df_historico.index[0]
        
        # Verifica se os valores são numéricos válidos antes de plotar
        if pd.notna(preco_justo):
            fig.add_trace(go.Scatter(
                x=[primeiro_indice, ultimo_indice],
                y=[preco_justo, preco_justo],
                mode='lines',
                name='Preço Justo',
                line=dict(color='green', width=1, dash='dash')
            ))
        
        if pd.notna(preco_compra_forte):
            fig.add_trace(go.Scatter(
                x=[primeiro_indice, ultimo_indice],
                y=[preco_compra_forte, preco_compra_forte],
                mode='lines',
                name='Compra Forte',
                line=dict(color='darkgreen', width=1, dash='dot')
            ))
    
    # Configurar layout
    fig.update_layout(
        title=f'Histórico de Preços - {ticker}',
        xaxis_title='Data',
        yaxis_title='Preço (R$)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    
    return fig

# Função para criar gráfico de radar (sem grandes mudanças, mas verifica NaNs)
def criar_grafico_radar(df_resumo, ticker, setor):
    # ... (código anterior com tratamento de NaNs e erros) ...
    if df_resumo is None or df_resumo.empty or setor is None:
        return None
        
    # Filtrar ações do mesmo setor
    df_setor = df_resumo[df_resumo['Setor'] == setor].copy()
    
    if len(df_setor) <= 1:
        return None
    
    # Métricas para o radar
    metricas = ['Preço Atual', 'Preço Justo', 'Potencial']
    
    # Verificar se as colunas existem
    for metrica in metricas:
        if metrica not in df_setor.columns:
             st.warning(f"Métrica '{metrica}' não encontrada no resumo para o gráfico de radar.")
             return None
             
    # Converter para numérico e remover NaNs/Infs apenas para as métricas
    df_norm = df_setor.copy()
    for metrica in metricas:
        df_norm[metrica] = pd.to_numeric(df_norm[metrica], errors='coerce')
    df_norm.dropna(subset=metricas, inplace=True)
    df_norm = df_norm[np.isfinite(df_norm[metricas]).all(axis=1)]

    if df_norm.empty or len(df_norm) <= 1:
        st.info(f"Dados insuficientes ou inválidos no setor '{setor}' para gerar radar após limpeza.")
        return None

    # Normalizar métricas para comparação
    metricas_norm_cols = []
    for metrica in metricas:
        col_norm = f'{metrica}_norm'
        metricas_norm_cols.append(col_norm)
        max_val = df_norm[metrica].max()
        min_val = df_norm[metrica].min()
        if max_val != min_val and pd.notna(max_val) and pd.notna(min_val):
            df_norm[col_norm] = (df_norm[metrica] - min_val) / (max_val - min_val)
        else:
            df_norm[col_norm] = 0.5 # Valor padrão se não houver variação ou dados inválidos
    
    fig = go.Figure()
    
    # Adicionar cada ação como um traço no radar
    for _, acao in df_norm.iterrows():
        # Usa as colunas normalizadas para os valores 'r'
        valores = acao[metricas_norm_cols].tolist()
        simbolo_acao = acao['Símbolo']
        
        # Destacar a ação selecionada
        if simbolo_acao == ticker:
            largura = 3
            opacidade = 1.0
        else:
            largura = 1
            opacidade = 0.7
        
        fig.add_trace(go.Scatterpolar(
            r=valores,
            theta=metricas, # Usa nomes originais das métricas no eixo theta
            fill='toself',
            name=simbolo_acao,
            line=dict(width=largura),
            opacity=opacidade
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] # Eixo normalizado
            )
        ),
        title=f'Comparação Setorial - {setor}',
        showlegend=True,
        height=400
    )
    
    return fig

# Função para criar mapa de calor (sem grandes mudanças, mas verifica NaNs)
def criar_mapa_calor(df_resumo):
    # ... (código anterior com tratamento de NaNs e erros) ...
    if df_resumo is None or df_resumo.empty or 'Recomendação' not in df_resumo.columns or 'Setor' not in df_resumo.columns:
        st.warning("Dados insuficientes no resumo para gerar o mapa de calor.")
        return go.Figure(), pd.DataFrame()
        
    # Mapear recomendações para valores numéricos
    mapa_recomendacao = {
        'Compra Forte': 5, 'Compra': 4, 'Neutro': 3,
        'Venda Parcial': 2, 'Venda': 1
    }
    
    df_mapa = df_resumo.copy()
    # Garante que a coluna existe antes de mapear
    if 'Recomendação' in df_mapa.columns:
        df_mapa['Valor_Recomendacao'] = df_mapa['Recomendação'].map(mapa_recomendacao).fillna(0) # Trata NaNs e não mapeados
    else:
        st.warning("Coluna 'Recomendação' não encontrada para o mapa de calor.")
        return go.Figure(), pd.DataFrame()

    # Agrupar por setor e calcular média, tratando NaNs
    if 'Setor' in df_mapa.columns:
        df_setor = df_mapa.groupby('Setor')['Valor_Recomendacao'].mean().reset_index()
        df_setor.dropna(subset=['Valor_Recomendacao'], inplace=True) # Remove setores sem recomendação média válida
    else:
        st.warning("Coluna 'Setor' não encontrada para o mapa de calor.")
        return go.Figure(), pd.DataFrame()
        
    if df_setor.empty:
        st.info("Nenhum dado de setor válido para o mapa de calor após agrupamento.")
        return go.Figure(), df_setor

    # Define a recomendação média baseada no valor
    def get_recomendacao_media(x):
        if pd.isna(x): return 'Indefinido'
        if x >= 4.5: return 'Compra Forte'
        if x >= 3.5: return 'Compra'
        if x >= 2.5: return 'Neutro'
        if x >= 1.5: return 'Venda Parcial'
        if x > 0: return 'Venda'
        return 'Indefinido'
        
    df_setor['Recomendação_Média'] = df_setor['Valor_Recomendacao'].apply(get_recomendacao_media)
    
    # Criar mapa de calor
    try:
        # Ordena setores para melhor visualização (opcional)
        df_setor_sorted = df_setor.sort_values('Valor_Recomendacao', ascending=False)
        
        pivot_table = df_setor_sorted.set_index('Setor')[['Valor_Recomendacao']]
        
        if pivot_table.empty:
            st.warning("Não foi possível gerar a tabela pivot para o mapa de calor.")
            return go.Figure(), df_setor
            
        fig = px.imshow(
            pivot_table.values,
            labels=dict(x="", y="Setor", color="Score Médio"),
            y=pivot_table.index,
            x=['Recomendação Média'], # Label do eixo x
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[1, 5], # Define a escala de cor baseada nos valores mapeados
            text_auto='.2f', # Formata o texto para duas casas decimais
            aspect="auto" # Ajusta o aspecto
        )
        
        fig.update_xaxes(showticklabels=False) # Esconde ticks do eixo x
        fig.update_layout(
            title='Recomendação Média por Setor',
            height=max(300, len(pivot_table.index) * 25 + 50), # Ajusta altura dinamicamente
            coloraxis_colorbar=dict(title="Score") # Título da barra de cor
        )
    except Exception as e:
        st.error(f"Erro ao criar o mapa de calor: {e}")
        return go.Figure(), df_setor
    
    return fig, df_setor_sorted # Retorna o df ordenado

# Função para exibir detalhes da ação (adaptada)
def exibir_detalhes_acao(dados_completos, ticker_sa, df_resumo):
    # ... (código anterior, mas busca dados de 'dados_completos' e usa converter_historico_para_df) ...
    ticker = ticker_sa.replace('.SA', '') # Usa o ticker sem .SA para exibição
    
    if dados_completos is None or ticker_sa not in dados_completos:
        st.error(f"Dados não disponíveis para {ticker_sa}. Tente atualizar os dados.")
        return
    
    acao_data = dados_completos[ticker_sa]
    info = acao_data.get('info', {})
    valuation = acao_data.get('valuation', {})
    insights = acao_data.get('insights', {})
    
    # Converter histórico (que está serializado) para DataFrame
    df_historico = converter_historico_para_df(acao_data.get('historico', {}))
    
    # Layout em colunas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Informações básicas
        st.subheader(f"{info.get('name', 'Nome Indisponível')} ({ticker})")
        st.write(f"**Setor:** {valuation.get('setor', 'N/A')}")
        st.write(f"**Método de Valuation:** {valuation.get('metodo_valuation', 'N/A')}")
        
        # Gráfico de preços
        fig_precos = criar_grafico_precos(
            df_historico, 
            ticker, 
            valuation.get('preco_justo'), 
            valuation.get('preco_compra_forte')
        )
        st.plotly_chart(fig_precos, use_container_width=True)
    
    with col2:
        # Card de recomendação
        recomendacao = valuation.get('recomendacao', 'Indefinido')
        cor_recomendacao = {
            'Compra Forte': 'darkgreen', 'Compra': 'green', 'Neutro': 'gray',
            'Venda Parcial': 'orange', 'Venda': 'red'
        }.get(recomendacao, 'black')
        
        preco_atual = valuation.get('preco_atual', np.nan)
        preco_justo = valuation.get('preco_justo', np.nan)
        preco_compra_forte = valuation.get('preco_compra_forte', np.nan)
        
        potencial_str = "N/A"
        if pd.notna(preco_atual) and pd.notna(preco_justo) and preco_atual > 0:
            potencial = ((preco_justo / preco_atual - 1) * 100)
            potencial_str = f"{potencial:.2f}%"
            
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
            <h4 style="text-align: center; margin-bottom: 5px;">Recomendação</h4>
            <h3 style="text-align: center; color: {cor_recomendacao}; margin-top: 0px; margin-bottom: 10px;">{recomendacao}</h3>
            <hr style="margin-top: 5px; margin-bottom: 10px;">
            <p style="font-size: 0.9em; margin-bottom: 3px;"><strong>Preço Atual:</strong> R$ {preco_atual:.2f if pd.notna(preco_atual) else 'N/A'}</p>
            <p style="font-size: 0.9em; margin-bottom: 3px;"><strong>Preço Justo:</strong> R$ {preco_justo:.2f if pd.notna(preco_justo) else 'N/A'}</p>
            <p style="font-size: 0.9em; margin-bottom: 3px;"><strong>Compra Forte:</strong> R$ {preco_compra_forte:.2f if pd.notna(preco_compra_forte) else 'N/A'}</p>
            <p style="font-size: 0.9em; margin-bottom: 0px;"><strong>Potencial:</strong> {potencial_str}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Insights técnicos
        if insights and 'technical' in insights and insights['technical']:
            st.subheader("Análise Técnica")
            tech_data = []
            tech_info = insights['technical']
            
            for term, key in [("Curto Prazo", 'short_term'), ("Médio Prazo", 'mid_term'), ("Longo Prazo", 'long_term')]:
                if key in tech_info and tech_info[key]: # Verifica se a chave e o valor existem
                    direction = tech_info[key].get('direction', 'N/A')
                    score = tech_info[key].get('score', 'N/A')
                    tech_data.append([term, direction, score])
            
            if tech_data:
                df_tech = pd.DataFrame(tech_data, columns=["Prazo", "Direção", "Score"])
                
                def highlight_direction(val):
                    if val == 'up': return 'background-color: #c6efce; color: #006100'
                    if val == 'down': return 'background-color: #ffc7ce; color: #9c0006'
                    return ''
                
                st.dataframe(
                    df_tech.style.applymap(highlight_direction, subset=['Direção']),
                    hide_index=True,
                    use_container_width=True
                )
            else:
                st.info("Nenhuma análise técnica disponível.")
        else:
             st.info("Nenhuma análise técnica disponível.")

    # Seção de análise comparativa
    st.subheader("Análise Comparativa Setorial")
    setor_acao = valuation.get('setor')
    if setor_acao and df_resumo is not None and not df_resumo.empty:
        col1_comp, col2_comp = st.columns([1, 2])
        
        with col1_comp:
            radar_fig = criar_grafico_radar(df_resumo, ticker, setor_acao)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info(f"Dados insuficientes no setor '{setor_acao}' para comparação via radar.")
        
        with col2_comp:
            df_setor_comp = df_resumo[df_resumo['Setor'] == setor_acao].copy()
            
            if not df_setor_comp.empty:
                def highlight_row(row):
                    if row['Símbolo'] == ticker:
                        return ['background-color: #e6f2ff'] * len(row)
                    return [''] * len(row)
                
                cols_to_show = ['Símbolo', 'Nome', 'Preço Atual', 'Preço Justo', 'Potencial', 'Recomendação']
                # Garante que as colunas existem antes de tentar acessá-las
                cols_present = [col for col in cols_to_show if col in df_setor_comp.columns]
                df_display_comp = df_setor_comp[cols_present].copy()
                
                # Formatação segura
                if 'Potencial' in df_display_comp.columns:
                    df_display_comp['Potencial'] = df_display_comp['Potencial'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
                if 'Preço Atual' in df_display_comp.columns:
                    df_display_comp['Preço Atual'] = df_display_comp['Preço Atual'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')
                if 'Preço Justo' in df_display_comp.columns:
                    df_display_comp['Preço Justo'] = df_display_comp['Preço Justo'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')

                st.dataframe(
                    df_display_comp.style.apply(highlight_row, axis=1),
                    hide_index=True,
                    use_container_width=True,
                    height=min(400, len(df_display_comp)*35 + 40) # Altura dinâmica
                )
            else:
                 st.info(f"Nenhuma outra ação encontrada no setor '{setor_acao}' para comparação.")
    else:
        st.warning("Setor da ação não definido ou resumo de dados indisponível para comparação.")

# Função para exibir visão geral (adaptada)
def exibir_visao_geral(df_resumo):
    # ... (código anterior com tratamento de NaNs e erros) ...
    if df_resumo is None or df_resumo.empty:
        st.warning("Resumo de dados indisponível para exibir a visão geral. Tente atualizar os dados.")
        return
        
    st.subheader("Visão Geral do Mercado")
    
    # Mapa de calor e tabela de recomendação por setor
    col1_vg, col2_vg = st.columns([2, 1])
    with col1_vg:
        mapa_calor, df_setor_mapa = criar_mapa_calor(df_resumo)
        if mapa_calor:
             st.plotly_chart(mapa_calor, use_container_width=True)
        else:
             st.warning("Não foi possível gerar o mapa de calor de recomendação por setor.")
    
    with col2_vg:
        if not df_setor_mapa.empty:
            st.subheader("Recomendação Média") # Título mais curto
            def highlight_recomendacao(val):
                colors = {
                    'Compra Forte': 'background-color: #006100; color: white',
                    'Compra': 'background-color: #c6efce; color: #006100',
                    'Neutro': 'background-color: #ffeb9c; color: #9c6500',
                    'Venda Parcial': 'background-color: #ffcc99; color: #9c3400',
                    'Venda': 'background-color: #ffc7ce; color: #9c0006'
                }
                return colors.get(val, '')
            
            st.dataframe(
                df_setor_mapa[['Setor', 'Recomendação_Média']].style.applymap(
                    highlight_recomendacao, subset=['Recomendação_Média']
                ),
                hide_index=True,
                use_container_width=True,
                height=min(400, len(df_setor_mapa)*35 + 40) # Altura dinâmica
            )
        else:
            st.info("Dados de recomendação por setor indisponíveis.")
            
    st.divider()
    
    # Distribuição de recomendações e Top 5 Potencial
    st.subheader("Distribuição e Destaques")
    col1_dist, col2_dist = st.columns([1, 2])
    with col1_dist:
        st.markdown("**Distribuição de Recomendações**")
        if 'Recomendação' in df_resumo.columns:
            recomendacoes_count = df_resumo['Recomendação'].value_counts().reset_index()
            recomendacoes_count.columns = ['Recomendação', 'Contagem']
            ordem = ['Compra Forte', 'Compra', 'Neutro', 'Venda Parcial', 'Venda', 'Indefinido']
            recomendacoes_count['ordem'] = pd.Categorical(recomendacoes_count['Recomendação'], categories=ordem, ordered=True)
            recomendacoes_count = recomendacoes_count.sort_values('ordem').drop('ordem', axis=1)
            
            cores = {
                'Compra Forte': 'darkgreen', 'Compra': 'green', 'Neutro': 'gray',
                'Venda Parcial': 'orange', 'Venda': 'red', 'Indefinido': 'black'
            }
            
            fig_dist = px.bar(
                recomendacoes_count,
                x='Recomendação', y='Contagem', color='Recomendação',
                color_discrete_map=cores, text='Contagem'
            )
            fig_dist.update_layout(showlegend=False, height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Dados de recomendação indisponíveis.")
            
    with col2_dist:
        st.markdown("**Top 5 Ações por Potencial**")
        if 'Potencial' in df_resumo.columns:
            df_resumo_pot = df_resumo.copy()
            df_resumo_pot['Potencial_Num'] = pd.to_numeric(df_resumo_pot['Potencial'], errors='coerce')
            # Ordena por potencial descendente, tratando NaNs
            top_potencial = df_resumo_pot.sort_values('Potencial_Num', ascending=False, na_position='last').head(5)
            
            cols_top = ['Símbolo', 'Nome', 'Potencial', 'Preço Atual', 'Preço Justo', 'Recomendação']
            cols_present_top = [col for col in cols_top if col in top_potencial.columns]
            df_display_top = top_potencial[cols_present_top].copy()

            # Formatação segura
            if 'Potencial' in df_display_top.columns:
                 df_display_top['Potencial'] = df_display_top['Potencial'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
            if 'Preço Atual' in df_display_top.columns:
                 df_display_top['Preço Atual'] = df_display_top['Preço Atual'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')
            if 'Preço Justo' in df_display_top.columns:
                 df_display_top['Preço Justo'] = df_display_top['Preço Justo'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')

            st.dataframe(
                df_display_top,
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Dados de potencial indisponíveis.")
            
    st.divider()
    
    # Tabela completa com todas as ações
    st.subheader("Tabela Completa de Ações")
    def highlight_recomendacao_row(row):
        rec = row.get('Recomendação', '')
        if rec == 'Compra Forte': return ['background-color: #c6efce'] * len(row)
        if rec == 'Compra': return ['background-color: #e6f2ff'] * len(row)
        if rec == 'Venda': return ['background-color: #ffc7ce'] * len(row)
        return [''] * len(row)
    
    df_display_all = df_resumo.copy()
    # Formatação segura
    if 'Potencial' in df_display_all.columns:
        df_display_all['Potencial'] = df_display_all['Potencial'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
    if 'Preço Atual' in df_display_all.columns:
        df_display_all['Preço Atual'] = df_display_all['Preço Atual'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')
    if 'Preço Justo' in df_display_all.columns:
        df_display_all['Preço Justo'] = df_display_all['Preço Justo'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')
    if 'Preço Compra Forte' in df_display_all.columns:
         df_display_all['Preço Compra Forte'] = df_display_all['Preço Compra Forte'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')

    cols_all = ['Símbolo', 'Nome', 'Setor', 'Preço Atual', 'Preço Justo', 'Potencial', 'Recomendação', 'Método']
    cols_present_all = [col for col in cols_all if col in df_display_all.columns]
    df_display_final = df_display_all[cols_present_all]

    st.dataframe(
        df_display_final.style.apply(highlight_recomendacao_row, axis=1),
        hide_index=True,
        use_container_width=True
    )

# Função para exibir metodologia (sem mudanças)
def exibir_metodologia():
    # ... (código anterior) ...
    st.subheader("Metodologia de Análise e Recomendação")
    
    st.markdown("""
    ### Filosofia de Investimento
    
    O modelo de análise e recomendação de ações implementado neste painel é baseado na filosofia de investimento do fundo SPX FALCON MASTER FI, que adota uma abordagem híbrida combinando:
    
    - **Análise Fundamentalista (Stock-picking)**: Seleção de empresas com base em seus fundamentos e valor intrínseco.
    - **Análise de Mercado (Market Timing)**: Avaliação de momentos de mercado e ajuste da exposição ao risco.
    
    Esta abordagem integrada permite identificar oportunidades de investimento considerando tanto o valor das empresas quanto o contexto macroeconômico.
    
    ### Métodos Quantitativos
    
    O modelo utiliza diversos métodos quantitativos para análise e recomendação:
    
    1.  **Modelos de Valuation Setoriais (Simplificados)**:
        *   A valuation é calculada com base em heurísticas e ajustes técnicos sobre o preço atual, considerando parâmetros médios por setor (como P/L justo implícito). A implementação ideal exigiria acesso a mais dados fundamentalistas (LPA, VPA, Receita, EBITDA) que não estão disponíveis via API básica utilizada.
        *   Os métodos listados (Gordon Growth, FCD, Múltiplos) representam a *intenção* da análise setorial, mas a execução atual é simplificada.
    
    2.  **Análise Técnica**:
        *   Tendências de curto, médio e longo prazo (quando disponíveis via API).
        *   Scores técnicos são usados para aplicar um fator de ajuste ao cálculo do preço justo.
    
    3.  **Modelo de Recomendação**:
        *   **Compra Forte**: Preço atual < Preço Justo * 0.80
        *   **Compra**: Preço Justo * 0.80 <= Preço atual < Preço Justo * 0.95
        *   **Neutro**: Preço Justo * 0.95 <= Preço atual <= Preço Justo * 1.05
        *   **Venda Parcial**: Preço Justo * 1.05 < Preço atual <= Preço Justo * 1.15
        *   **Venda**: Preço atual > Preço Justo * 1.15

    ### Fontes de Dados
    
    Os dados utilizados neste painel são coletados via API da `YahooFinance` (através da `data_api` do Manus) no momento da atualização.
    
    *Disclaimer: Esta análise é gerada por um modelo automatizado com simplificações e não constitui recomendação de investimento personalizada. Consulte um profissional financeiro antes de tomar decisões.*
    """)

# --- Interface Principal --- 
st.title("📈 Análise de Ações Brasileiras - Modelo SPX FALCON (v. Integrada)")

# --- Gerenciamento de Estado e Botão de Atualização --- 

# Inicializa o estado da sessão se não existir
if 'dados_acoes' not in st.session_state:
    st.session_state['dados_acoes'] = None
if 'resumo_acoes' not in st.session_state:
    st.session_state['resumo_acoes'] = None
if 'ultima_atualizacao' not in st.session_state:
    st.session_state['ultima_atualizacao'] = None

st.sidebar.title("Controles")

# Botão para forçar a atualização
if st.sidebar.button("Buscar/Atualizar Dados Agora", key="update_button"):
    with st.spinner("Atualizando todos os dados... Isso pode levar alguns minutos."):
        # Chama a função de atualização (o cache será ignorado devido à chamada explícita)
        # Para realmente forçar, podemos limpar o cache específico ou usar um argumento extra
        # Mas chamar diretamente aqui deve buscar dados se o cache expirou ou é a primeira vez.
        # Para garantir a atualização *sempre* que o botão é clicado, limpamos o cache:
        atualizar_dados_completos.clear()
        try:
            dados_atualizados, resumo_atualizado = atualizar_dados_completos(acoes_default)
            st.session_state['dados_acoes'] = dados_atualizados
            st.session_state['resumo_acoes'] = resumo_atualizado
            st.session_state['ultima_atualizacao'] = datetime.now()
            st.sidebar.success("Dados atualizados com sucesso!")
            st.rerun() # Força o recarregamento da UI com os novos dados
        except Exception as e:
            st.sidebar.error(f"Erro durante a atualização: {e}")
            # Mantém os dados antigos no estado, se houver

# Carrega os dados do estado da sessão se já existirem
dados_carregados = st.session_state['dados_acoes']
resumo_carregado = st.session_state['resumo_acoes']

# Se os dados ainda não foram carregados (primeira execução ou após erro na atualização manual)
if dados_carregados is None or resumo_carregado is None:
    st.warning("Dados ainda não carregados. Clique em 'Buscar/Atualizar Dados Agora' na barra lateral.")
    # Opcional: Tentar carregar automaticamente na primeira vez
    # with st.spinner("Carregando dados iniciais..."):
    #     dados_carregados, resumo_carregado = atualizar_dados_completos(acoes_default)
    #     st.session_state['dados_acoes'] = dados_carregados
    #     st.session_state['resumo_acoes'] = resumo_carregado
    #     st.session_state['ultima_atualizacao'] = datetime.now()
    #     st.rerun()

# Exibe a data da última atualização
if st.session_state['ultima_atualizacao']:
    st.sidebar.caption(f"Última atualização: {st.session_state['ultima_atualizacao'].strftime('%d/%m/%Y %H:%M:%S')}")
else:
    st.sidebar.caption("Dados ainda não atualizados.")

# --- Exibição Principal --- 

# Verifica se os dados foram carregados antes de tentar exibir
if dados_carregados is not None and resumo_carregado is not None:
    # Opções de visualização na sidebar
    st.sidebar.header("Visualização")
    
    # Tenta obter a lista de ações dos dados carregados
    lista_acoes_disponiveis = sorted(list(dados_carregados.keys()))
    nomes_formatados = []
    for ticker_sa in lista_acoes_disponiveis:
         nome = dados_carregados[ticker_sa].get('info', {}).get('name', ticker_sa.replace('.SA', ''))
         nomes_formatados.append(f"{nome} ({ticker_sa.replace('.SA', '')})")
         
    # Mapeamento reverso para obter o ticker_sa a partir do nome formatado
    mapa_nome_ticker = {nome_fmt: ticker for nome_fmt, ticker in zip(nomes_formatados, lista_acoes_disponiveis)}

    # Define as opções de visualização
    opcoes_view = ["Visão Geral do Mercado", "Detalhes por Ação", "Metodologia"]
    
    # Seleção da visualização
    # Usamos um selectbox em vez de radio para melhor visual com muitas ações
    view_selecionada = st.sidebar.selectbox("Escolha a visualização:", opcoes_view, key="view_select")

    if view_selecionada == "Visão Geral do Mercado":
        exibir_visao_geral(resumo_carregado)
        
    elif view_selecionada == "Detalhes por Ação":
        if lista_acoes_disponiveis:
            # Usar selectbox para a ação
            nome_acao_selecionada = st.selectbox(
                "Selecione uma Ação:", 
                options=nomes_formatados, # Usa os nomes formatados
                index=0, # Padrão para a primeira ação
                key="stock_select"
            )
            ticker_selecionado_sa = mapa_nome_ticker.get(nome_acao_selecionada)
            
            if ticker_selecionado_sa:
                exibir_detalhes_acao(dados_carregados, ticker_selecionado_sa, resumo_carregado)
            else:
                st.error("Erro ao encontrar o ticker para a ação selecionada.")
        else:
            st.warning("Nenhuma ação disponível para seleção. Atualize os dados.")
            
    elif view_selecionada == "Metodologia":
        exibir_metodologia()
else:
    # Mensagem se os dados não puderam ser carregados
    st.error("Não foi possível carregar ou atualizar os dados. Verifique a conexão ou tente atualizar novamente.")

