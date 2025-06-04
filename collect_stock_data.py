import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import time

# Criar diretório para armazenar os dados
os.makedirs('data', exist_ok=True)

# Lista de ações solicitadas
acoes = [
    'AGRO3.SA', 'BBAS3.SA', 'BBSE3.SA', 'BPAC11.SA', 'EGIE3.SA', 'ITUB3.SA', 
    'PRIO3.SA', 'PSSA3.SA', 'SAPR3.SA', 'SBSP3.SA', 'VIVT3.SA', 'WEGE3.SA', 
    'TOTS3.SA', 'B3SA3.SA', 'TAEE3.SA', 'CMIG3.SA',
    # Outras ações principais do Ibovespa
    'PETR4.SA', 'VALE3.SA', 'ITSA4.SA', 'ABEV3.SA', 'RENT3.SA', 'MGLU3.SA',
    'SUZB3.SA', 'RADL3.SA', 'CSAN3.SA', 'EQTL3.SA'
]

# Inicializar cliente da API
client = ApiClient()

# Função para coletar dados históricos de preços
def coletar_dados_historicos(simbolo):
    print(f"Coletando dados históricos para {simbolo}...")
    try:
        # Coleta dados de 5 anos
        dados = client.call_api('YahooFinance/get_stock_chart', query={
            'symbol': simbolo,
            'interval': '1mo',
            'range': '5y',
            'region': 'BR',
            'includeAdjustedClose': True
        })
        
        # Verificar se há resultados
        if not dados or 'chart' not in dados or 'result' not in dados['chart'] or not dados['chart']['result']:
            print(f"Sem dados para {simbolo}")
            return None
        
        # Extrair dados
        result = dados['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        # Criar DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': quotes['open'],
            'high': quotes['high'],
            'low': quotes['low'],
            'close': quotes['close'],
            'volume': quotes['volume']
        })
        
        # Converter timestamp para data
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('date', inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        
        # Adicionar informações da empresa
        meta = result['meta']
        info = {
            'symbol': meta.get('symbol', ''),
            'name': meta.get('shortName', ''),
            'currency': meta.get('currency', ''),
            'exchange': meta.get('exchangeName', ''),
            'current_price': meta.get('regularMarketPrice', 0),
            '52w_high': meta.get('fiftyTwoWeekHigh', 0),
            '52w_low': meta.get('fiftyTwoWeekLow', 0)
        }
        
        return {'data': df, 'info': info}
    
    except Exception as e:
        print(f"Erro ao coletar dados para {simbolo}: {e}")
        return None

# Função para coletar insights e valuation
def coletar_insights(simbolo):
    print(f"Coletando insights para {simbolo}...")
    try:
        insights = client.call_api('YahooFinance/get_stock_insights', query={
            'symbol': simbolo
        })
        
        if not insights or 'finance' not in insights or 'result' not in insights['finance']:
            print(f"Sem insights para {simbolo}")
            return None
        
        result = insights['finance']['result']
        
        # Extrair dados de valuation e recomendação
        valuation_data = {}
        if 'instrumentInfo' in result and 'valuation' in result['instrumentInfo']:
            valuation = result['instrumentInfo']['valuation']
            valuation_data = {
                'description': valuation.get('description', ''),
                'discount': valuation.get('discount', ''),
                'relative_value': valuation.get('relativeValue', '')
            }
        
        # Extrair recomendação
        recommendation = {}
        if 'recommendation' in result:
            rec = result['recommendation']
            recommendation = {
                'target_price': rec.get('targetPrice', 0),
                'rating': rec.get('rating', '')
            }
        
        # Extrair eventos técnicos
        technical_events = {}
        if 'instrumentInfo' in result and 'technicalEvents' in result['instrumentInfo']:
            tech = result['instrumentInfo']['technicalEvents']
            if 'shortTermOutlook' in tech:
                technical_events['short_term'] = {
                    'direction': tech['shortTermOutlook'].get('direction', ''),
                    'score': tech['shortTermOutlook'].get('score', 0),
                    'description': tech['shortTermOutlook'].get('scoreDescription', '')
                }
            if 'intermediateTermOutlook' in tech:
                technical_events['mid_term'] = {
                    'direction': tech['intermediateTermOutlook'].get('direction', ''),
                    'score': tech['intermediateTermOutlook'].get('score', 0),
                    'description': tech['intermediateTermOutlook'].get('scoreDescription', '')
                }
            if 'longTermOutlook' in tech:
                technical_events['long_term'] = {
                    'direction': tech['longTermOutlook'].get('direction', ''),
                    'score': tech['longTermOutlook'].get('score', 0),
                    'description': tech['longTermOutlook'].get('scoreDescription', '')
                }
        
        return {
            'valuation': valuation_data,
            'recommendation': recommendation,
            'technical': technical_events
        }
    
    except Exception as e:
        print(f"Erro ao coletar insights para {simbolo}: {e}")
        return None

# Definir parâmetros setoriais para valuation
setores = {
    'Financeiro': {
        'acoes': ['BBAS3.SA', 'ITUB3.SA', 'B3SA3.SA', 'BPAC11.SA', 'BBSE3.SA', 'PSSA3.SA'],
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
        'acoes': ['AGRO3.SA', 'PRIO3.SA', 'VALE3.SA', 'SUZB3.SA', 'CSAN3.SA'],
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

# Função para calcular preço justo baseado no setor
def calcular_preco_justo(simbolo, preco_atual, insights):
    # Encontrar o setor da ação
    setor = None
    for nome_setor, info in setores.items():
        if simbolo in info['acoes']:
            setor = nome_setor
            break
    
    if not setor:
        # Se não encontrou setor específico, usa parâmetros genéricos
        setor = 'Geral'
        parametros = {
            'crescimento_longo_prazo': 0.05,
            'premio_risco': 0.06,
            'beta_medio': 1.0,
            'p_l_justo': 14
        }
        metodo = 'Múltiplos Gerais'
    else:
        parametros = setores[setor]['parametros']
        metodo = setores[setor]['metodo']
    
    # Ajustar com base nos insights, se disponíveis
    ajuste = 1.0
    if insights and 'technical' in insights:
        # Usar tendência de longo prazo para ajustar o preço justo
        if 'long_term' in insights['technical']:
            direction = insights['technical']['long_term'].get('direction', '')
            score = insights['technical']['long_term'].get('score', 0)
            
            if direction == 'up':
                ajuste = 1.0 + (score / 10)
            elif direction == 'down':
                ajuste = 1.0 - (score / 10)
    
    # Calcular preço justo baseado no método do setor
    if metodo == 'Gordon Growth':
        # Para financeiras, usamos P/VP ajustado
        preco_justo = preco_atual * parametros['p_vp_justo'] / 1.5 * ajuste
    elif metodo == 'Fluxo de Caixa Descontado':
        # Para utilities, usamos dividend yield
        preco_justo = preco_atual * (0.05 / parametros['dividend_yield_justo']) * ajuste
    elif metodo == 'Múltiplos de Receita':
        # Para tecnologia, usamos PS ratio
        preco_justo = preco_atual * parametros['ps_ratio_justo'] / 3.0 * ajuste
    elif metodo == 'Ciclo de Preços':
        # Para commodities, usamos EV/EBITDA
        preco_justo = preco_atual * parametros['ev_ebitda_justo'] / 5.0 * ajuste
    else:
        # Método genérico baseado em P/L
        preco_justo = preco_atual * parametros['p_l_justo'] / 12 * ajuste
    
    # Calcular preço de compra forte (20% abaixo do preço justo)
    preco_compra_forte = preco_justo * 0.8
    
    # Determinar recomendação
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
        'setor': setor,
        'metodo_valuation': metodo,
        'ajuste_tecnico': ajuste
    }

# Função para converter DataFrame para formato serializável em JSON
def df_to_serializable(df):
    if df is None:
        return None
    
    # Converter índice de data para string
    result = {}
    for date, row in df.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        result[date_str] = row.to_dict()
    
    return result

# Coletar e processar dados para todas as ações
resultados = {}

for acao in acoes:
    print(f"\nProcessando {acao}...")
    
    # Coletar dados históricos
    dados_historicos = coletar_dados_historicos(acao)
    if not dados_historicos:
        continue
    
    # Coletar insights
    insights = coletar_insights(acao)
    
    # Calcular preço justo
    preco_atual = dados_historicos['info']['current_price']
    valuation = calcular_preco_justo(acao, preco_atual, insights)
    
    # Armazenar resultados (convertendo DataFrame para formato serializável)
    resultados[acao] = {
        'historico': df_to_serializable(dados_historicos['data']),
        'info': dados_historicos['info'],
        'insights': insights,
        'valuation': valuation
    }
    
    # Aguardar um pouco para não sobrecarregar a API
    time.sleep(1)

# Salvar resultados em arquivo JSON
with open('acoes_dados.json', 'w') as f:
    json.dump(resultados, f)

print("\nDados coletados e salvos com sucesso!")

# Criar um DataFrame resumido para visualização rápida
resumo = []
for simbolo, dados in resultados.items():
    resumo.append({
        'Símbolo': simbolo.replace('.SA', ''),
        'Nome': dados['info']['name'],
        'Preço Atual': dados['valuation']['preco_atual'],
        'Preço Justo': round(dados['valuation']['preco_justo'], 2),
        'Preço Compra Forte': round(dados['valuation']['preco_compra_forte'], 2),
        'Potencial': round((dados['valuation']['preco_justo'] / dados['valuation']['preco_atual'] - 1) * 100, 2),
        'Recomendação': dados['valuation']['recomendacao'],
        'Setor': dados['valuation']['setor'],
        'Método': dados['valuation']['metodo_valuation']
    })

df_resumo = pd.DataFrame(resumo)
df_resumo.to_csv("resumo_acoes.csv", index=False)
print("Resumo salvo em data/resumo_acoes.csv")
