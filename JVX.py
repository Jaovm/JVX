import pandas as pd
import numpy as np
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import subprocess
import sys

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
        # Define os caminhos dos arquivos de dados
        script_dir = os.path.dirname(__file__)
        json_path = os.path.join(script_dir, 'acoes_dados.json')
        csv_path = os.path.join(script_dir, 'resumo_acoes.csv')

        # Verifica se os arquivos existem
        if not os.path.exists(json_path) or not os.path.exists(csv_path):
            st.warning("Arquivos de dados não encontrados. Execute a atualização inicial.")
            return None, None

        # Carregar dados das ações
        with open(json_path, 'r') as f:
            dados = json.load(f)
        
        # Carregar resumo
        resumo = pd.read_csv(csv_path)
        
        return dados, resumo
    except FileNotFoundError:
        st.error("Erro: Arquivos de dados (acoes_dados.json ou resumo_acoes.csv) não encontrados. Execute a atualização.")
        return None, None
    except json.JSONDecodeError:
        st.error("Erro ao decodificar o arquivo JSON (acoes_dados.json). O arquivo pode estar corrompido ou vazio.")
        return None, None
    except pd.errors.EmptyDataError:
        st.error("Erro: O arquivo CSV (resumo_acoes.csv) está vazio.")
        return None, None
    except Exception as e:
        st.error(f"Erro inesperado ao carregar dados: {e}")
        return None, None

# --- Sidebar --- 
st.sidebar.title("Controles")
if st.sidebar.button("Buscar e Atualizar Dados"): 
    with st.spinner("Atualizando dados... Aguarde, isso pode levar alguns minutos."):
        try:
            # Define o diretório de trabalho como o diretório do script atual
            script_dir = os.path.dirname(__file__)
            collect_script_path = os.path.join(script_dir, "collect_stock_data.py")
            
            # Verifica se o script de coleta existe
            if not os.path.exists(collect_script_path):
                st.sidebar.error(f"Erro: Script 'collect_stock_data.py' não encontrado em {script_dir}")
            else:
                # Executa o script de coleta de dados
                process = subprocess.run(
                    [sys.executable, collect_script_path],
                    check=True, 
                    capture_output=True, 
                    text=True,
                    cwd=script_dir # Garante que o script execute no diretório correto
                )
                st.sidebar.success("Dados atualizados com sucesso!")
                # Limpa o cache para forçar o recarregamento na próxima execução
                st.cache_data.clear()
                # Força o rerun para garantir a atualização imediata da UI
                st.rerun()
                
        except subprocess.CalledProcessError as e:
            st.sidebar.error(f"Erro ao executar o script de atualização:")
            st.sidebar.code(f"Comando: {e.cmd}\nRetorno: {e.returncode}\nOutput:\n{e.stdout}\nErro:\n{e.stderr}")
        except Exception as e:
            st.sidebar.error(f"Erro inesperado durante a atualização: {e}")

# --- Carregar Dados --- 
# Chama a função para carregar os dados (será recarregada se o cache foi limpo)
dados, resumo = carregar_dados()

# Função para converter dados históricos de volta para DataFrame
def converter_historico_para_df(historico_dict):
    if not historico_dict:
        return pd.DataFrame()
    
    dados_list = []
    for data_str, valores in historico_dict.items():
        linha = valores.copy()
        try:
            linha['date'] = datetime.strptime(data_str, '%Y-%m-%d')
            dados_list.append(linha)
        except (ValueError, TypeError):
            # Ignora entradas mal formatadas ou nulas
            continue 
            
    if not dados_list:
        return pd.DataFrame()
        
    df = pd.DataFrame(dados_list)
    df.set_index('date', inplace=True)
    # Garante que as colunas numéricas sejam numéricas, tratando erros
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Função para criar gráfico de preços
def criar_grafico_precos(df_historico, ticker, preco_justo, preco_compra_forte):
    if df_historico.empty or 'close' not in df_historico.columns:
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

# Função para criar gráfico de radar para comparação setorial
def criar_grafico_radar(df_resumo, ticker, setor):
    if df_resumo is None or setor is None:
        return None
        
    # Filtrar ações do mesmo setor
    df_setor = df_resumo[df_resumo['Setor'] == setor].copy()
    
    if len(df_setor) <= 1:
        return None
    
    # Normalizar métricas para comparação
    metricas = ['Preço Atual', 'Preço Justo', 'Potencial']
    df_norm = df_setor.copy()
    
    # Tratar valores ausentes ou infinitos antes da normalização
    for metrica in metricas:
        if metrica not in df_norm.columns:
             st.warning(f"Métrica '{metrica}' não encontrada no resumo para o gráfico de radar.")
             return None
        df_norm[metrica] = pd.to_numeric(df_norm[metrica], errors='coerce')
        df_norm.dropna(subset=[metrica], inplace=True)
        df_norm = df_norm[np.isfinite(df_norm[metrica])]

    if df_norm.empty:
        return None

    for metrica in metricas:
        max_val = df_norm[metrica].max()
        min_val = df_norm[metrica].min()
        if max_val != min_val and pd.notna(max_val) and pd.notna(min_val):
            df_norm[f'{metrica}_norm'] = (df_norm[metrica] - min_val) / (max_val - min_val)
        else:
            df_norm[f'{metrica}_norm'] = 0.5 # Valor padrão se não houver variação ou dados inválidos
    
    # Criar dados para o gráfico radar
    categorias = df_norm['Símbolo'].tolist()
    metricas_norm = [f'{m}_norm' for m in metricas]
    
    fig = go.Figure()
    
    # Adicionar cada ação como um traço no radar
    for _, acao in df_norm.iterrows():
        valores = acao[metricas_norm].tolist()
        
        # Destacar a ação selecionada
        if acao['Símbolo'] == ticker:
            largura = 3
            opacidade = 1.0
        else:
            largura = 1
            opacidade = 0.7
        
        fig.add_trace(go.Scatterpolar(
            r=valores,
            theta=metricas, # Usar nomes originais das métricas no eixo
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
    if df_resumo is None or 'Recomendação' not in df_resumo.columns or 'Setor' not in df_resumo.columns:
        st.warning("Dados insuficientes no resumo para gerar o mapa de calor.")
        return go.Figure(), pd.DataFrame()
        
    # Mapear recomendações para valores numéricos
    mapa_recomendacao = {
        'Compra Forte': 5,
        'Compra': 4,
        'Neutro': 3,
        'Venda Parcial': 2,
        'Venda': 1
    }
    
    df_mapa = df_resumo.copy()
    df_mapa['Valor_Recomendacao'] = df_mapa['Recomendação'].map(mapa_recomendacao).fillna(0) # Trata recomendações não mapeadas
    
    # Agrupar por setor e calcular média
    df_setor = df_mapa.groupby('Setor')['Valor_Recomendacao'].mean().reset_index()
    
    # Define a recomendação média baseada no valor
    def get_recomendacao_media(x):
        if x >= 4.5: return 'Compra Forte'
        if x >= 3.5: return 'Compra'
        if x >= 2.5: return 'Neutro'
        if x >= 1.5: return 'Venda Parcial'
        if x > 0: return 'Venda'
        return 'Indefinido' # Caso a média seja 0 (sem dados válidos)
        
    df_setor['Recomendação_Média'] = df_setor['Valor_Recomendacao'].apply(get_recomendacao_media)
    
    # Criar mapa de calor
    try:
        pivot_table = df_setor.pivot_table(index='Setor', values='Valor_Recomendacao', aggfunc='mean')
        if pivot_table.empty:
            st.warning("Não foi possível gerar a tabela pivot para o mapa de calor.")
            return go.Figure(), df_setor
            
        fig = px.imshow(
            pivot_table.values,
            labels=dict(x="", y="Setor", color="Score Médio"),
            y=pivot_table.index,
            color_continuous_scale=['red', 'yellow', 'green'],
            range_color=[1, 5], # Define a escala de cor baseada nos valores mapeados
            text_auto='.2f' # Formata o texto para duas casas decimais
        )
        
        fig.update_layout(
            title='Recomendação Média por Setor',
            height=max(300, len(pivot_table.index) * 30) # Ajusta altura dinamicamente
        )
    except Exception as e:
        st.error(f"Erro ao criar o mapa de calor: {e}")
        return go.Figure(), df_setor
    
    return fig, df_setor

# Função para exibir detalhes da ação
def exibir_detalhes_acao(dados, ticker_sa, df_resumo):
    ticker = ticker_sa.replace('.SA', '') # Usa o ticker sem .SA para exibição
    
    if dados is None or ticker_sa not in dados:
        st.error(f"Dados não disponíveis para {ticker_sa}. Tente atualizar os dados.")
        return
    
    acao_data = dados[ticker_sa]
    info = acao_data.get('info', {})
    valuation = acao_data.get('valuation', {})
    insights = acao_data.get('insights', {})
    
    # Converter histórico para DataFrame
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
            'Compra Forte': 'darkgreen',
            'Compra': 'green',
            'Neutro': 'gray',
            'Venda Parcial': 'orange',
            'Venda': 'red'
        }.get(recomendacao, 'black')
        
        preco_atual = valuation.get('preco_atual', np.nan)
        preco_justo = valuation.get('preco_justo', np.nan)
        preco_compra_forte = valuation.get('preco_compra_forte', np.nan)
        
        potencial_str = "N/A"
        if pd.notna(preco_atual) and pd.notna(preco_justo) and preco_atual != 0:
            potencial = ((preco_justo / preco_atual - 1) * 100)
            potencial_str = f"{potencial:.2f}%"
            
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="text-align: center;">Recomendação</h3>
            <h2 style="text-align: center; color: {cor_recomendacao};">{recomendacao}</h2>
            <hr>
            <p><strong>Preço Atual:</strong> R$ {preco_atual:.2f if pd.notna(preco_atual) else 'N/A'}</p>
            <p><strong>Preço Justo:</strong> R$ {preco_justo:.2f if pd.notna(preco_justo) else 'N/A'}</p>
            <p><strong>Preço Compra Forte:</strong> R$ {preco_compra_forte:.2f if pd.notna(preco_compra_forte) else 'N/A'}</p>
            <p><strong>Potencial:</strong> {potencial_str}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Insights técnicos
        if insights and 'technical' in insights:
            st.subheader("Análise Técnica")
            tech_data = []
            tech_info = insights['technical']
            
            for term, key in [("Curto Prazo", 'short_term'), ("Médio Prazo", 'mid_term'), ("Longo Prazo", 'long_term')]:
                if key in tech_info and tech_info[key]:
                    direction = tech_info[key].get('direction', 'N/A')
                    score = tech_info[key].get('score', 'N/A')
                    tech_data.append([term, direction, score])
            
            if tech_data:
                df_tech = pd.DataFrame(tech_data, columns=["Prazo", "Direção", "Score"])
                
                # Formatar cores baseadas na direção
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
    if setor_acao and df_resumo is not None:
        col1_comp, col2_comp = st.columns([1, 2])
        
        with col1_comp:
            # Gráfico de radar para comparação setorial
            radar_fig = criar_grafico_radar(df_resumo, ticker, setor_acao)
            if radar_fig:
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info(f"Não há dados suficientes no setor '{setor_acao}' para comparação via radar.")
        
        with col2_comp:
            # Tabela de comparação com outras ações do mesmo setor
            df_setor_comp = df_resumo[df_resumo['Setor'] == setor_acao].copy()
            
            if not df_setor_comp.empty:
                # Destacar a ação atual
                def highlight_row(row):
                    if row['Símbolo'] == ticker:
                        return ['background-color: #e6f2ff'] * len(row)
                    return [''] * len(row)
                
                # Selecionar e formatar colunas para exibição
                cols_to_show = ['Símbolo', 'Nome', 'Preço Atual', 'Preço Justo', 'Potencial', 'Recomendação']
                df_display_comp = df_setor_comp[cols_to_show].copy()
                df_display_comp['Potencial'] = df_display_comp['Potencial'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
                df_display_comp['Preço Atual'] = df_display_comp['Preço Atual'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')
                df_display_comp['Preço Justo'] = df_display_comp['Preço Justo'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')

                st.dataframe(
                    df_display_comp.style.apply(highlight_row, axis=1),
                    hide_index=True,
                    use_container_width=True
                )
            else:
                 st.info(f"Nenhuma outra ação encontrada no setor '{setor_acao}' para comparação.")
    else:
        st.warning("Setor da ação não definido ou resumo de dados indisponível para comparação.")

# Função para exibir visão geral de todas as ações
def exibir_visao_geral(df_resumo):
    if df_resumo is None:
        st.warning("Resumo de dados indisponível para exibir a visão geral.")
        return
        
    st.subheader("Visão Geral do Mercado")
    
    # Mapa de calor de recomendações por setor
    col1_vg, col2_vg = st.columns([2, 1])
    
    with col1_vg:
        mapa_calor, df_setor = criar_mapa_calor(df_resumo)
        if mapa_calor:
             st.plotly_chart(mapa_calor, use_container_width=True)
        else:
             st.warning("Não foi possível gerar o mapa de calor.")
    
    with col2_vg:
        if not df_setor.empty:
            st.subheader("Recomendação Média por Setor")
            
            # Formatar cores baseadas na recomendação
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
                df_setor[['Setor', 'Recomendação_Média']].style.applymap(
                    highlight_recomendacao, subset=['Recomendação_Média']
                ),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Dados de recomendação por setor indisponíveis.")
            
    # Distribuição de recomendações
    st.subheader("Distribuição de Recomendações")
    
    col1_dist, col2_dist = st.columns([1, 2])
    
    with col1_dist:
        if 'Recomendação' in df_resumo.columns:
            # Contagem de recomendações
            recomendacoes_count = df_resumo['Recomendação'].value_counts().reset_index()
            recomendacoes_count.columns = ['Recomendação', 'Contagem']
            
            # Ordenar por nível de recomendação
            ordem = ['Compra Forte', 'Compra', 'Neutro', 'Venda Parcial', 'Venda']
            recomendacoes_count['ordem'] = pd.Categorical(recomendacoes_count['Recomendação'], categories=ordem, ordered=True)
            recomendacoes_count = recomendacoes_count.sort_values('ordem').drop('ordem', axis=1)
            
            # Gráfico de barras
            cores = {
                'Compra Forte': 'darkgreen', 'Compra': 'green', 'Neutro': 'gray',
                'Venda Parcial': 'orange', 'Venda': 'red'
            }
            
            fig_dist = px.bar(
                recomendacoes_count,
                x='Recomendação',
                y='Contagem',
                color='Recomendação',
                color_discrete_map=cores,
                text='Contagem'
            )
            
            fig_dist.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("Dados de recomendação indisponíveis para o gráfico de distribuição.")
            
    with col2_dist:
        # Tabela de ações com maior potencial
        st.subheader("Top 5 Ações com Maior Potencial")
        if 'Potencial' in df_resumo.columns:
            # Garante que Potencial seja numérico para ordenação
            df_resumo['Potencial_Num'] = pd.to_numeric(df_resumo['Potencial'], errors='coerce')
            top_potencial = df_resumo.sort_values('Potencial_Num', ascending=False).head(5)
            
            # Formata colunas para exibição
            cols_top = ['Símbolo', 'Nome', 'Preço Atual', 'Preço Justo', 'Potencial', 'Recomendação', 'Setor']
            df_display_top = top_potencial[cols_top].copy()
            df_display_top['Potencial'] = df_display_top['Potencial'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
            df_display_top['Preço Atual'] = df_display_top['Preço Atual'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')
            df_display_top['Preço Justo'] = df_display_top['Preço Justo'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')

            st.dataframe(
                df_display_top,
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Dados de potencial indisponíveis para classificar as ações.")
            
    # Tabela completa com todas as ações
    st.subheader("Todas as Ações")
    
    # Adicionar formatação para potencial e recomendação
    def highlight_recomendacao_row(row):
        rec = row.get('Recomendação', '')
        if rec == 'Compra Forte': return ['background-color: #c6efce'] * len(row)
        if rec == 'Compra': return ['background-color: #e6f2ff'] * len(row)
        if rec == 'Venda': return ['background-color: #ffc7ce'] * len(row)
        return [''] * len(row)
    
    # Formatar potencial como percentual e preços como moeda
    df_display_all = df_resumo.copy()
    if 'Potencial' in df_display_all.columns:
        df_display_all['Potencial'] = df_display_all['Potencial'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
    if 'Preço Atual' in df_display_all.columns:
        df_display_all['Preço Atual'] = df_display_all['Preço Atual'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')
    if 'Preço Justo' in df_display_all.columns:
        df_display_all['Preço Justo'] = df_display_all['Preço Justo'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')
    if 'Preço Compra Forte' in df_display_all.columns:
         df_display_all['Preço Compra Forte'] = df_display_all['Preço Compra Forte'].apply(lambda x: f"R$ {x:.2f}" if pd.notna(x) else 'N/A')

    # Seleciona colunas relevantes para exibição
    cols_all = ['Símbolo', 'Nome', 'Preço Atual', 'Preço Justo', 'Preço Compra Forte', 'Potencial', 'Recomendação', 'Setor', 'Método']
    cols_present = [col for col in cols_all if col in df_display_all.columns]
    df_display_final = df_display_all[cols_present]

    st.dataframe(
        df_display_final.style.apply(highlight_recomendacao_row, axis=1),
        hide_index=True,
        use_container_width=True
    )

# Função para exibir metodologia
def exibir_metodologia():
    st.subheader("Metodologia de Análise e Recomendação")
    
    st.markdown("""
    ### Filosofia de Investimento
    
    O modelo de análise e recomendação de ações implementado neste painel é baseado na filosofia de investimento do fundo SPX FALCON MASTER FI, que adota uma abordagem híbrida combinando:
    
    - **Análise Fundamentalista (Stock-picking)**: Seleção de empresas com base em seus fundamentos e valor intrínseco.
    - **Análise de Mercado (Market Timing)**: Avaliação de momentos de mercado e ajuste da exposição ao risco.
    
    Esta abordagem integrada permite identificar oportunidades de investimento considerando tanto o valor das empresas quanto o contexto macroeconômico.
    
    ### Métodos Quantitativos
    
    O modelo utiliza diversos métodos quantitativos para análise e recomendação:
    
    1.  **Modelos de Valuation Setoriais**:
        *   Financeiro: Gordon Growth e P/VP ajustado.
        *   Utilities: Fluxo de Caixa Descontado com foco em dividend yield.
        *   Tecnologia: Múltiplos de Receita (PS Ratio).
        *   Commodities: Ciclo de Preços e EV/EBITDA.
        *   Consumo: Múltiplos Comparáveis com ajuste de crescimento.
    
    2.  **Análise Técnica**:
        *   Tendências de curto, médio e longo prazo.
        *   Scores técnicos para ajuste do preço justo.
        *   Indicadores de momentum e reversão à média.
    
    3.  **Modelo de Recomendação**:
        *   **Compra Forte**: Preço atual significativamente abaixo do preço de compra forte (geralmente >20% de desconto sobre o preço justo).
        *   **Compra**: Preço atual abaixo do preço justo, mas acima do nível de compra forte.
        *   **Neutro**: Preço atual próximo ao preço justo.
        *   **Venda Parcial**: Preço atual ligeiramente acima do preço justo.
        *   **Venda**: Preço atual significativamente acima do preço justo.

    ### Fontes de Dados
    
    Os dados utilizados neste painel são coletados de fontes públicas e APIs financeiras (como Yahoo Finance), processados e analisados pelos modelos descritos.
    
    *Disclaimer: Esta análise é gerada por um modelo automatizado e não constitui recomendação de investimento personalizada. Consulte um profissional financeiro antes de tomar decisões.*
    """)

# --- Interface Principal --- 
st.title("📈 Análise de Ações Brasileiras - Modelo SPX FALCON")

# Verifica se os dados foram carregados
if dados is None or resumo is None:
    st.error("Não foi possível carregar os dados iniciais. Por favor, clique em 'Buscar e Atualizar Dados' na barra lateral.")
else:
    # Opções de visualização na sidebar
    st.sidebar.header("Visualização")
    opcao_visualizacao = st.sidebar.radio(
        "Escolha a visualização:",
        ("Visão Geral do Mercado", "Detalhes por Ação", "Metodologia")
    )

    if opcao_visualizacao == "Visão Geral do Mercado":
        exibir_visao_geral(resumo)
        
    elif opcao_visualizacao == "Detalhes por Ação":
        # Selecionar ação
        lista_acoes = sorted(list(dados.keys()))
        ticker_selecionado = st.selectbox("Selecione uma Ação:", lista_acoes, format_func=lambda x: f"{dados[x]['info'].get('name', 'N/A')} ({x.replace('.SA', '')})")
        
        if ticker_selecionado:
            exibir_detalhes_acao(dados, ticker_selecionado, resumo)
            
    elif opcao_visualizacao == "Metodologia":
        exibir_metodologia()

