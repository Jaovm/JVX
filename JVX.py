import pandas as pd
import numpy as np
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Configuração da página
st.set_page_config(
    page_title="Análise de Ações Brasileiras - Modelo SPX FALCON",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para carregar dados
@st.cache_data
def carregar_dados():
    try:
        # Carregar dados das ações
        with open('data/acoes_dados.json', 'r') as f:
            dados = json.load(f)
        
        # Carregar resumo
        resumo = pd.read_csv('data/resumo_acoes.csv')
        
        return dados, resumo
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None, None

# Função para converter dados históricos de volta para DataFrame
def converter_historico_para_df(historico_dict):
    if not historico_dict:
        return pd.DataFrame()
    
    dados = []
    for data_str, valores in historico_dict.items():
        linha = valores.copy()
        linha['date'] = datetime.strptime(data_str, '%Y-%m-%d')
        dados.append(linha)
    
    df = pd.DataFrame(dados)
    df.set_index('date', inplace=True)
    return df

# Função para criar gráfico de preços
def criar_grafico_precos(df_historico, ticker, preco_justo, preco_compra_forte):
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
    ultimo_indice = df_historico.index[-1]
    primeiro_indice = df_historico.index[0]
    
    fig.add_trace(go.Scatter(
        x=[primeiro_indice, ultimo_indice],
        y=[preco_justo, preco_justo],
        mode='lines',
        name='Preço Justo',
        line=dict(color='green', width=1, dash='dash')
    ))
    
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

# Função para criar gráfico de radar para comparação setorial
def criar_grafico_radar(df_resumo, ticker, setor):
    # Filtrar ações do mesmo setor
    df_setor = df_resumo[df_resumo['Setor'] == setor].copy()
    
    if len(df_setor) <= 1:
        return None
    
    # Normalizar métricas para comparação
    metricas = ['Preço Atual', 'Preço Justo', 'Potencial']
    df_norm = df_setor.copy()
    
    for metrica in metricas:
        max_val = df_setor[metrica].max()
        min_val = df_setor[metrica].min()
        if max_val != min_val:
            df_norm[f'{metrica}_norm'] = (df_setor[metrica] - min_val) / (max_val - min_val)
        else:
            df_norm[f'{metrica}_norm'] = 0.5
    
    # Criar dados para o gráfico radar
    categorias = df_norm['Símbolo'].tolist()
    
    fig = go.Figure()
    
    # Adicionar cada ação como um traço no radar
    for _, acao in df_norm.iterrows():
        valores = [acao['Preço Atual_norm'], acao['Preço Justo_norm'], acao['Potencial_norm']]
        
        # Destacar a ação selecionada
        if acao['Símbolo'] == ticker.replace('.SA', ''):
            largura = 3
            opacidade = 1.0
        else:
            largura = 1
            opacidade = 0.7
        
        fig.add_trace(go.Scatterpolar(
            r=valores,
            theta=metricas,
            fill='toself',
            name=acao['Símbolo'],
            line=dict(width=largura),
            opacity=opacidade
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title=f'Comparação Setorial - {setor}',
        showlegend=True,
        height=400
    )
    
    return fig

# Função para criar mapa de calor de recomendações
def criar_mapa_calor(df_resumo):
    # Mapear recomendações para valores numéricos
    mapa_recomendacao = {
        'Compra Forte': 5,
        'Compra': 4,
        'Neutro': 3,
        'Venda Parcial': 2,
        'Venda': 1
    }
    
    df_mapa = df_resumo.copy()
    df_mapa['Valor_Recomendacao'] = df_mapa['Recomendação'].map(mapa_recomendacao)
    
    # Agrupar por setor e calcular média
    df_setor = df_mapa.groupby('Setor')['Valor_Recomendacao'].mean().reset_index()
    df_setor['Recomendação_Média'] = df_setor['Valor_Recomendacao'].apply(
        lambda x: 'Compra Forte' if x >= 4.5 else 
                 ('Compra' if x >= 3.5 else 
                 ('Neutro' if x >= 2.5 else 
                 ('Venda Parcial' if x >= 1.5 else 'Venda')))
    )
    
    # Criar mapa de calor
    fig = px.imshow(
        df_setor.pivot_table(index='Setor', values='Valor_Recomendacao', aggfunc='mean').values,
        labels=dict(x="", y="Setor", color="Score"),
        y=df_setor['Setor'],
        color_continuous_scale=['red', 'yellow', 'green'],
        text_auto=True
    )
    
    fig.update_layout(
        title='Recomendação Média por Setor',
        height=300
    )
    
    return fig, df_setor

# Função para exibir detalhes da ação
def exibir_detalhes_acao(dados, ticker, df_resumo):
    if ticker not in dados:
        st.error(f"Dados não disponíveis para {ticker}")
        return
    
    info = dados[ticker]['info']
    valuation = dados[ticker]['valuation']
    insights = dados[ticker]['insights']
    
    # Converter histórico para DataFrame
    df_historico = converter_historico_para_df(dados[ticker]['historico'])
    
    # Layout em colunas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Informações básicas
        st.subheader(f"{info['name']} ({ticker.replace('.SA', '')})")
        st.write(f"**Setor:** {valuation['setor']}")
        st.write(f"**Método de Valuation:** {valuation['metodo_valuation']}")
        
        # Gráfico de preços
        fig_precos = criar_grafico_precos(
            df_historico, 
            ticker.replace('.SA', ''), 
            valuation['preco_justo'], 
            valuation['preco_compra_forte']
        )
        st.plotly_chart(fig_precos, use_container_width=True)
    
    with col2:
        # Card de recomendação
        recomendacao = valuation['recomendacao']
        cor_recomendacao = {
            'Compra Forte': 'darkgreen',
            'Compra': 'green',
            'Neutro': 'gray',
            'Venda Parcial': 'orange',
            'Venda': 'red'
        }.get(recomendacao, 'gray')
        
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="text-align: center;">Recomendação</h3>
            <h2 style="text-align: center; color: {cor_recomendacao};">{recomendacao}</h2>
            <hr>
            <p><strong>Preço Atual:</strong> R$ {valuation['preco_atual']:.2f}</p>
            <p><strong>Preço Justo:</strong> R$ {valuation['preco_justo']:.2f}</p>
            <p><strong>Preço Compra Forte:</strong> R$ {valuation['preco_compra_forte']:.2f}</p>
            <p><strong>Potencial:</strong> {((valuation['preco_justo'] / valuation['preco_atual'] - 1) * 100):.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Insights técnicos
        if insights and 'technical' in insights:
            st.subheader("Análise Técnica")
            
            tech_data = []
            if 'short_term' in insights['technical']:
                direction = insights['technical']['short_term']['direction']
                score = insights['technical']['short_term']['score']
                tech_data.append(["Curto Prazo", direction, score])
            
            if 'mid_term' in insights['technical']:
                direction = insights['technical']['mid_term']['direction']
                score = insights['technical']['mid_term']['score']
                tech_data.append(["Médio Prazo", direction, score])
            
            if 'long_term' in insights['technical']:
                direction = insights['technical']['long_term']['direction']
                score = insights['technical']['long_term']['score']
                tech_data.append(["Longo Prazo", direction, score])
            
            if tech_data:
                df_tech = pd.DataFrame(tech_data, columns=["Prazo", "Direção", "Score"])
                
                # Formatar cores baseadas na direção
                def highlight_direction(val):
                    if val == 'up':
                        return 'background-color: #c6efce; color: #006100'
                    elif val == 'down':
                        return 'background-color: #ffc7ce; color: #9c0006'
                    return ''
                
                st.dataframe(
                    df_tech.style.applymap(highlight_direction, subset=['Direção']),
                    hide_index=True
                )
    
    # Seção de análise comparativa
    st.subheader("Análise Comparativa Setorial")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Gráfico de radar para comparação setorial
        radar_fig = criar_grafico_radar(df_resumo, ticker.replace('.SA', ''), valuation['setor'])
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info(f"Não há ações suficientes no setor {valuation['setor']} para comparação.")
    
    with col2:
        # Tabela de comparação com outras ações do mesmo setor
        df_setor = df_resumo[df_resumo['Setor'] == valuation['setor']].copy()
        
        # Destacar a ação atual
        def highlight_row(row):
            if row['Símbolo'] == ticker.replace('.SA', ''):
                return ['background-color: #e6f2ff'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            df_setor.style.apply(highlight_row, axis=1),
            hide_index=True,
            use_container_width=True
        )

# Função para exibir visão geral de todas as ações
def exibir_visao_geral(df_resumo):
    st.subheader("Visão Geral do Mercado")
    
    # Mapa de calor de recomendações por setor
    col1, col2 = st.columns([2, 1])
    
    with col1:
        mapa_calor, df_setor = criar_mapa_calor(df_resumo)
        st.plotly_chart(mapa_calor, use_container_width=True)
    
    with col2:
        st.subheader("Recomendação por Setor")
        
        # Formatar cores baseadas na recomendação
        def highlight_recomendacao(val):
            if val == 'Compra Forte':
                return 'background-color: #006100; color: white'
            elif val == 'Compra':
                return 'background-color: #c6efce; color: #006100'
            elif val == 'Neutro':
                return 'background-color: #ffeb9c; color: #9c6500'
            elif val == 'Venda Parcial':
                return 'background-color: #ffcc99; color: #9c3400'
            elif val == 'Venda':
                return 'background-color: #ffc7ce; color: #9c0006'
            return ''
        
        st.dataframe(
            df_setor[['Setor', 'Recomendação_Média']].style.applymap(
                highlight_recomendacao, subset=['Recomendação_Média']
            ),
            hide_index=True,
            use_container_width=True
        )
    
    # Distribuição de recomendações
    st.subheader("Distribuição de Recomendações")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Contagem de recomendações
        recomendacoes_count = df_resumo['Recomendação'].value_counts().reset_index()
        recomendacoes_count.columns = ['Recomendação', 'Contagem']
        
        # Ordenar por nível de recomendação
        ordem = ['Compra Forte', 'Compra', 'Neutro', 'Venda Parcial', 'Venda']
        recomendacoes_count['ordem'] = recomendacoes_count['Recomendação'].apply(lambda x: ordem.index(x) if x in ordem else 999)
        recomendacoes_count = recomendacoes_count.sort_values('ordem').drop('ordem', axis=1)
        
        # Gráfico de barras
        cores = {
            'Compra Forte': 'darkgreen',
            'Compra': 'green',
            'Neutro': 'gray',
            'Venda Parcial': 'orange',
            'Venda': 'red'
        }
        
        fig = px.bar(
            recomendacoes_count,
            x='Recomendação',
            y='Contagem',
            color='Recomendação',
            color_discrete_map=cores,
            text='Contagem'
        )
        
        fig.update_layout(
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tabela de ações com maior potencial
        st.subheader("Top 5 Ações com Maior Potencial")
        top_potencial = df_resumo.sort_values('Potencial', ascending=False).head(5)
        
        def format_potencial(val):
            return f"{val:.2f}%"
        
        top_potencial['Potencial_fmt'] = top_potencial['Potencial'].apply(format_potencial)
        
        st.dataframe(
            top_potencial[['Símbolo', 'Nome', 'Preço Atual', 'Preço Justo', 'Potencial_fmt', 'Recomendação', 'Setor']],
            hide_index=True,
            use_container_width=True
        )
    
    # Tabela completa com todas as ações
    st.subheader("Todas as Ações")
    
    # Adicionar formatação para potencial e recomendação
    def highlight_recomendacao_row(row):
        rec = row['Recomendação']
        if rec == 'Compra Forte':
            return ['background-color: #c6efce'] * len(row)
        elif rec == 'Compra':
            return ['background-color: #e6f2ff'] * len(row)
        elif rec == 'Venda':
            return ['background-color: #ffc7ce'] * len(row)
        return [''] * len(row)
    
    # Formatar potencial como percentual
    df_display = df_resumo.copy()
    df_display['Potencial'] = df_display['Potencial'].apply(lambda x: f"{x:.2f}%")
    
    st.dataframe(
        df_display.style.apply(highlight_recomendacao_row, axis=1),
        hide_index=True,
        use_container_width=True
    )

# Função para exibir metodologia
def exibir_metodologia():
    st.subheader("Metodologia de Análise e Recomendação")
    
    st.markdown("""
    ### Filosofia de Investimento
    
    O modelo de análise e recomendação de ações implementado neste painel é baseado na filosofia de investimento do fundo SPX FALCON MASTER FI, que adota uma abordagem híbrida combinando:
    
    - **Análise Fundamentalista (Stock-picking)**: Seleção de empresas com base em seus fundamentos e valor intrínseco
    - **Análise de Mercado (Market Timing)**: Avaliação de momentos de mercado e ajuste da exposição ao risco
    
    Esta abordagem integrada permite identificar oportunidades de investimento considerando tanto o valor das empresas quanto o contexto macroeconômico.
    
    ### Métodos Quantitativos
    
    O modelo utiliza diversos métodos quantitativos para análise e recomendação:
    
    1. **Modelos de Valuation Setoriais**:
       - Financeiro: Gordon Growth e P/VP ajustado
       - Utilities: Fluxo de Caixa Descontado com foco em dividend yield
       - Tecnologia: Múltiplos de Receita (PS Ratio)
       - Commodities: Ciclo de Preços e EV/EBITDA
       - Consumo: Múltiplos Comparáveis com ajuste de crescimento
    
    2. **Análise Técnica**:
       - Tendências de curto, médio e longo prazo
       - Scores técnicos para ajuste do preço justo
       - Indicadores de momentum e reversão à média
    
    3. **Gestão de Risco**:
       - Comparação setorial para identificar anomalias
       - Análise de potencial de valorização vs. risco
    
    ### Critérios de Recomendação
    
    As recomendações são baseadas na relação entre o preço atual e o preço justo calculado:
    
    - **Compra Forte**: Preço atual < 80% do preço justo
    - **Compra**: Preço atual entre 80% e 95% do preço justo
    - **Neutro**: Preço atual entre 95% e 105% do preço justo
    - **Venda Parcial**: Preço atual entre 105% e 115% do preço justo
    - **Venda**: Preço atual > 115% do preço justo
    
    ### Limitações do Modelo
    
    - O modelo é uma simplificação da abordagem real utilizada por gestores profissionais
    - As recomendações são baseadas em dados históricos e podem não refletir eventos futuros
    - Os parâmetros setoriais são estimativas e podem variar conforme condições de mercado
    - Este painel tem finalidade educacional e não constitui recomendação formal de investimento
    """)

# Função principal
def main():
    # Carregar dados
    dados, df_resumo = carregar_dados()
    
    if dados is None or df_resumo is None:
        st.error("Não foi possível carregar os dados. Verifique se os arquivos de dados existem.")
        return
    
    # Sidebar
    st.sidebar.title("Análise de Ações Brasileiras")
    st.sidebar.subheader("Modelo SPX FALCON")
    
    # Opções de navegação
    pagina = st.sidebar.radio(
        "Navegação",
        ["Visão Geral", "Análise Individual", "Metodologia"]
    )
    
    # Filtro de ações para análise individual
    if pagina == "Análise Individual":
        # Extrair tickers disponíveis
        tickers = list(dados.keys())
        tickers_sem_sa = [t.replace('.SA', '') for t in tickers]
        
        # Dropdown para seleção de ação
        ticker_selecionado = st.sidebar.selectbox(
            "Selecione uma ação",
            tickers_sem_sa
        )
        
        # Converter ticker selecionado de volta para formato com .SA
        ticker_completo = f"{ticker_selecionado}.SA"
    
    # Filtros adicionais para visão geral
    if pagina == "Visão Geral":
        st.sidebar.subheader("Filtros")
        
        # Filtro de setor
        setores = sorted(df_resumo['Setor'].unique())
        setor_selecionado = st.sidebar.multiselect(
            "Setor",
            setores,
            default=[]
        )
        
        # Filtro de recomendação
        recomendacoes = sorted(df_resumo['Recomendação'].unique())
        recomendacao_selecionada = st.sidebar.multiselect(
            "Recomendação",
            recomendacoes,
            default=[]
        )
        
        # Aplicar filtros
        if setor_selecionado:
            df_resumo = df_resumo[df_resumo['Setor'].isin(setor_selecionado)]
        
        if recomendacao_selecionada:
            df_resumo = df_resumo[df_resumo['Recomendação'].isin(recomendacao_selecionada)]
    
    # Exibir conteúdo com base na navegação
    if pagina == "Visão Geral":
        st.title("Visão Geral do Mercado")
        st.write("Análise comparativa de todas as ações monitoradas")
        
        exibir_visao_geral(df_resumo)
    
    elif pagina == "Análise Individual":
        st.title(f"Análise Individual - {ticker_selecionado}")
        
        exibir_detalhes_acao(dados, ticker_completo, df_resumo)
    
    elif pagina == "Metodologia":
        st.title("Metodologia")
        
        exibir_metodologia()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 0.8em;">
        Desenvolvido com base na filosofia de investimento do SPX FALCON MASTER FI | Dados atualizados em: Maio/2025
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
