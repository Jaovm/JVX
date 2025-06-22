# valuation_models.py

import pandas as pd
import numpy as np
import yfinance as yf

def safe_float(value, default=np.nan):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def get_financial_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    # Tenta buscar demonstrações financeiras anuais e trimestrais
    try:
        financials = ticker.financials
        quarterly_financials = ticker.quarterly_financials
    except Exception:
        financials = pd.DataFrame()
        quarterly_financials = pd.DataFrame()

    # Tenta buscar balanço patrimonial
    try:
        balance_sheet = ticker.balance_sheet
        quarterly_balance_sheet = ticker.quarterly_balance_sheet
    except Exception:
        balance_sheet = pd.DataFrame()
        quarterly_balance_sheet = pd.DataFrame()

    # Tenta buscar fluxo de caixa
    try:
        cash_flow = ticker.cashflow
        quarterly_cash_flow = ticker.quarterly_cashflow
    except Exception:
        cash_flow = pd.DataFrame()
        quarterly_cash_flow = pd.DataFrame()

    # Tenta buscar dividendos
    try:
        dividends = ticker.dividends
    except Exception:
        dividends = pd.Series()

    return {
        'financials': financials,
        'quarterly_financials': quarterly_financials,
        'balance_sheet': balance_sheet,
        'quarterly_balance_sheet': quarterly_balance_sheet,
        'cash_flow': cash_flow,
        'quarterly_cash_flow': quarterly_cash_flow,
        'dividends': dividends
    }

# --- Modelos de Múltiplos ---

def calculate_pe(info):
    trailing_pe = safe_float(info.get('trailingPE'))
    if pd.isna(trailing_pe) or trailing_pe <= 0:
        return np.nan
    return trailing_pe

def calculate_pb(info):
    price_to_book = safe_float(info.get('priceToBook'))
    if pd.isna(price_to_book) or price_to_book <= 0:
        return np.nan
    return price_to_book

def calculate_ps(info):
    price_to_sales = safe_float(info.get('priceToSales'))
    if pd.isna(price_to_sales) or price_to_sales <= 0:
        return np.nan
    return price_to_sales

def calculate_ev_ebitda(info):
    enterprise_value = safe_float(info.get('enterpriseValue'))
    ebitda = safe_float(info.get('ebitda'))
    if pd.isna(enterprise_value) or pd.isna(ebitda) or ebitda <= 0:
        return np.nan
    return enterprise_value / ebitda

def calculate_ev_ebit(info, financials):
    enterprise_value = safe_float(info.get('enterpriseValue'))
    # EBIT geralmente está em 'EBIT' ou 'OperatingIncome' nas demonstrações financeiras
    if not financials.empty and 'EBIT' in financials.index:
        ebit = safe_float(financials.loc['EBIT'].iloc[0])
    elif not financials.empty and 'OperatingIncome' in financials.index:
        ebit = safe_float(financials.loc['OperatingIncome'].iloc[0])
    else:
        ebit = np.nan

    if pd.isna(enterprise_value) or pd.isna(ebit) or ebit <= 0:
        return np.nan
    return enterprise_value / ebit

def calculate_ev_revenue(info, financials):
    enterprise_value = safe_float(info.get('enterpriseValue'))
    if not financials.empty and 'TotalRevenue' in financials.index:
        revenue = safe_float(financials.loc['TotalRevenue'].iloc[0])
    else:
        revenue = np.nan

    if pd.isna(enterprise_value) or pd.isna(revenue) or revenue <= 0:
        return np.nan
    return enterprise_value / revenue

# --- Modelos de Dividendos ---

def ddm_stable_growth(info, dividends, discount_rate, growth_rate):
    if dividends.empty or discount_rate <= 0 or growth_rate >= discount_rate:
        return np.nan
    
    # Pega o último dividendo pago
    last_dividend = safe_float(dividends.iloc[0])
    if pd.isna(last_dividend) or last_dividend <= 0:
        return np.nan

    # D1 = D0 * (1 + g)
    d1 = last_dividend * (1 + growth_rate)
    # Preço = D1 / (r - g)
    price = d1 / (discount_rate - growth_rate)
    return price

def ddm_multi_stage(info, dividends, discount_rate, growth_rates, growth_periods):
    if dividends.empty or discount_rate <= 0 or not growth_rates or not growth_periods:
        return np.nan

    last_dividend = safe_float(dividends.iloc[0])
    if pd.isna(last_dividend) or last_dividend <= 0:
        return np.nan

    pv_dividends = 0
    current_dividend = last_dividend
    cumulative_discount = 1

    for i, (g, period) in enumerate(zip(growth_rates, growth_periods)):
        for _ in range(period):
            current_dividend *= (1 + g)
            cumulative_discount *= (1 + discount_rate)
            pv_dividends += current_dividend / cumulative_discount

    # Calcular o valor terminal no final do último período de crescimento
    # Assumimos que o último crescimento é o crescimento estável
    terminal_growth_rate = growth_rates[-1]
    if discount_rate <= terminal_growth_rate:
        return np.nan # Crescimento maior ou igual à taxa de desconto

    terminal_dividend = current_dividend * (1 + terminal_growth_rate)
    terminal_value = terminal_dividend / (discount_rate - terminal_growth_rate)
    
    # Descontar o valor terminal para o presente
    pv_terminal_value = terminal_value / cumulative_discount

    return pv_dividends + pv_terminal_value

# --- Modelos de Fluxo de Caixa Descontado (DCF) ---

def calculate_wacc(info):
    # Simplificação: para um cálculo mais preciso, precisaríamos de dados de dívida, patrimônio, taxa de juros, etc.
    # O yfinance fornece 'beta', que pode ser usado para estimar o custo do capital próprio (CAPM).
    # Para o custo da dívida, precisaríamos de informações sobre a dívida da empresa e sua taxa de juros.
    # Por enquanto, usaremos uma estimativa baseada em beta e um custo de capital próprio e de dívida padrão.
    
    beta = safe_float(info.get('beta'))
    if pd.isna(beta):
        beta = 1.0 # Beta padrão se não disponível

    # Taxa de retorno livre de risco (ex: Selic ou Treasury Yield)
    risk_free_rate = 0.10 # Exemplo: 10% (ajustar para o contexto brasileiro)
    # Prêmio de risco de mercado (ex: retorno esperado do Ibovespa - risk_free_rate)
    market_risk_premium = 0.06 # Exemplo: 6%

    # Custo do Capital Próprio (Ke) usando CAPM
    cost_of_equity = risk_free_rate + beta * market_risk_premium

    # Custo da Dívida (Kd) - Simplificado, idealmente buscaríamos taxas de juros da dívida da empresa
    cost_of_debt = 0.12 # Exemplo: 12%
    # Alíquota de imposto de renda corporativo (para custo da dívida pós-imposto)
    tax_rate = 0.34 # Exemplo: 34% no Brasil
    cost_of_debt_after_tax = cost_of_debt * (1 - tax_rate)

    # Proporção Dívida/Patrimônio (D/E) - Simplificado, idealmente buscaríamos do balanço
    # Para uma estimativa, podemos usar o marketCap para o Patrimônio e tentar estimar a dívida
    market_cap = safe_float(info.get('marketCap'))
    total_debt = safe_float(info.get('totalDebt')) # yfinance pode ter isso em 'info'

    if pd.isna(market_cap) or market_cap <= 0:
        return np.nan
    
    if pd.isna(total_debt) or total_debt <= 0:
        # Se não tiver totalDebt, faz uma estimativa ou usa uma proporção padrão
        # Isso é uma grande simplificação e deve ser melhorado com dados reais
        total_debt = market_cap * 0.5 # Exemplo: Dívida é 50% do valor de mercado

    total_capital = market_cap + total_debt
    if total_capital <= 0:
        return np.nan

    weight_equity = market_cap / total_capital
    weight_debt = total_debt / total_capital

    wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt_after_tax)
    return wacc

def project_fcf(financials, cash_flow, years, revenue_growth_rate, ebitda_margin, capex_as_pct_revenue, nwc_as_pct_revenue):
    # Projeção de Fluxo de Caixa Livre (FCF)
    # Necessita de dados históricos para basear as projeções
    if financials.empty or cash_flow.empty:
        return [np.nan] * years

    # Pegar os dados mais recentes
    latest_revenue = safe_float(financials.loc['TotalRevenue'].iloc[0]) if 'TotalRevenue' in financials.index else np.nan
    latest_ebitda = safe_float(financials.loc['EBITDA'].iloc[0]) if 'EBITDA' in financials.index else np.nan
    latest_capex = safe_float(cash_flow.loc['CapitalExpenditures'].iloc[0]) if 'CapitalExpenditures' in cash_flow.index else np.nan
    latest_nwc_change = safe_float(cash_flow.loc['ChangeInWorkingCapital'].iloc[0]) if 'ChangeInWorkingCapital' in cash_flow.index else np.nan

    if pd.isna(latest_revenue) or latest_revenue <= 0:
        return [np.nan] * years

    projected_fcf = []
    current_revenue = latest_revenue

    for i in range(years):
        current_revenue *= (1 + revenue_growth_rate)
        projected_ebitda = current_revenue * ebitda_margin
        projected_capex = current_revenue * capex_as_pct_revenue
        # Simplificação: assumimos que a mudança no capital de giro é uma % da mudança na receita
        projected_nwc_change = current_revenue * nwc_as_pct_revenue

        # FCF = EBITDA - CAPEX - Mudança no Capital de Giro (simplificado)
        fcf = projected_ebitda - abs(projected_capex) - projected_nwc_change
        projected_fcf.append(fcf)
    return projected_fcf

def calculate_terminal_value(last_fcf, wacc, perpetual_growth_rate):
    if pd.isna(last_fcf) or wacc <= 0 or perpetual_growth_rate >= wacc:
        return np.nan
    return last_fcf * (1 + perpetual_growth_rate) / (wacc - perpetual_growth_rate)

def dcf_model(info, financials, cash_flow, years, revenue_growth_rate, ebitda_margin, capex_as_pct_revenue, nwc_as_pct_revenue, perpetual_growth_rate):
    wacc = calculate_wacc(info)
    if pd.isna(wacc):
        return np.nan

    projected_fcf = project_fcf(financials, cash_flow, years, revenue_growth_rate, ebitda_margin, capex_as_pct_revenue, nwc_as_pct_revenue)
    if any(pd.isna(fcf) for fcf in projected_fcf):
        return np.nan

    pv_fcf = 0
    for i, fcf in enumerate(projected_fcf):
        pv_fcf += fcf / ((1 + wacc)**(i + 1))

    terminal_value = calculate_terminal_value(projected_fcf[-1], wacc, perpetual_growth_rate)
    if pd.isna(terminal_value):
        return np.nan

    pv_terminal_value = terminal_value / ((1 + wacc)**years)

    enterprise_value = pv_fcf + pv_terminal_value

    # Valor do Patrimônio = Valor da Empresa - Dívida Líquida
    total_debt = safe_float(info.get('totalDebt'))
    cash_and_equivalents = safe_float(info.get('totalCash')) # ou 'cash'

    if pd.isna(total_debt) or pd.isna(cash_and_equivalents):
        # Tenta buscar do balanço patrimonial se não estiver em info
        if not info.get('balance_sheet', pd.DataFrame()).empty:
            bs = info['balance_sheet']
            total_debt = safe_float(bs.loc['TotalDebt'].iloc[0]) if 'TotalDebt' in bs.index else total_debt
            cash_and_equivalents = safe_float(bs.loc['CashAndCashEquivalents'].iloc[0]) if 'CashAndCashEquivalents' in bs.index else cash_and_equivalents

    if pd.isna(total_debt):
        total_debt = 0 # Assume 0 se não encontrar
    if pd.isna(cash_and_equivalents):
        cash_and_equivalents = 0 # Assume 0 se não encontrar

    net_debt = total_debt - cash_and_equivalents
    equity_value = enterprise_value - net_debt

    shares_outstanding = safe_float(info.get('sharesOutstanding'))
    if pd.isna(shares_outstanding) or shares_outstanding <= 0:
        return np.nan

    price_per_share = equity_value / shares_outstanding
    return price_per_share

# --- Earnings Power Value (EPV) ---

def earnings_power_value(info, financials, balance_sheet, discount_rate):
    if financials.empty or balance_sheet.empty or discount_rate <= 0:
        return np.nan

    # Lucro Operacional (EBIT) - assumimos que é sustentável
    if 'EBIT' in financials.index:
        ebit = safe_float(financials.loc['EBIT'].iloc[0])
    elif 'OperatingIncome' in financials.index:
        ebit = safe_float(financials.loc['OperatingIncome'].iloc[0])
    else:
        ebit = np.nan

    if pd.isna(ebit) or ebit <= 0:
        return np.nan

    # Impostos sobre o lucro operacional
    tax_rate = 0.34 # Exemplo: 34%
    nopat = ebit * (1 - tax_rate)

    # Capital de Giro Líquido (NWC) - para ajustar o capital investido
    # Ativos Circulantes - Passivos Circulantes
    current_assets = safe_float(balance_sheet.loc['CurrentAssets'].iloc[0]) if 'CurrentAssets' in balance_sheet.index else np.nan
    current_liabilities = safe_float(balance_sheet.loc['CurrentLiabilities'].iloc[0]) if 'CurrentLiabilities' in balance_sheet.index else np.nan

    if pd.isna(current_assets) or pd.isna(current_liabilities):
        nwc = 0 # Simplificação se não disponível
    else:
        nwc = current_assets - current_liabilities

    # Capital Investido = Ativos Fixos Líquidos + Capital de Giro Líquido
    # Propriedade, Planta e Equipamento Líquido (PPE)
    ppe = safe_float(balance_sheet.loc['NetPPE'].iloc[0]) if 'NetPPE' in balance_sheet.index else np.nan

    if pd.isna(ppe):
        return np.nan

    invested_capital = ppe + nwc

    # Valor da Empresa = NOPAT / WACC (assumindo NOPAT sustentável e sem crescimento)
    # Usamos o WACC como taxa de desconto para o valor da empresa
    wacc = calculate_wacc(info)
    if pd.isna(wacc) or wacc <= 0:
        return np.nan

    enterprise_value = nopat / wacc

    # Valor do Patrimônio = Valor da Empresa - Dívida Líquida
    total_debt = safe_float(info.get('totalDebt'))
    cash_and_equivalents = safe_float(info.get('totalCash'))

    if pd.isna(total_debt):
        total_debt = 0
    if pd.isna(cash_and_equivalents):
        cash_and_equivalents = 0

    net_debt = total_debt - cash_and_equivalents
    equity_value = enterprise_value - net_debt

    shares_outstanding = safe_float(info.get('sharesOutstanding'))
    if pd.isna(shares_outstanding) or shares_outstanding <= 0:
        return np.nan

    price_per_share = equity_value / shares_outstanding
    return price_per_share


def run_all_valuations(ticker_symbol, info, financials_data):
    results = {}
    
    # Extrair dados financeiros para facilitar o acesso
    financials = financials_data.get('financials', pd.DataFrame())
    cash_flow = financials_data.get('cash_flow', pd.DataFrame())
    dividends = financials_data.get('dividends', pd.Series())
    balance_sheet = financials_data.get('balance_sheet', pd.DataFrame())

    # Parâmetros de exemplo para DCF e DDM (podem ser ajustados ou parametrizados)
    # Taxa de desconto para DDM (ex: custo de capital próprio)
    ddm_discount_rate = 0.12 # 12%
    # Taxas de crescimento para DDM
    ddm_growth_stable = 0.03 # 3% crescimento estável
    ddm_growth_multi_stage = [0.15, 0.10, 0.03] # Crescimento alto, médio, estável
    ddm_growth_periods = [3, 2, 99] # 3 anos com 15%, 2 anos com 10%, depois estável

    # Parâmetros para DCF
    dcf_revenue_growth_rate_5y = 0.08 # 8% ao ano para 5 anos
    dcf_revenue_growth_rate_10y = 0.06 # 6% ao ano para 10 anos
    dcf_ebitda_margin = 0.25 # 25% de margem EBITDA (ajustar conforme setor/empresa)
    dcf_capex_as_pct_revenue = 0.05 # 5% da receita em CAPEX
    dcf_nwc_as_pct_revenue = 0.02 # 2% da receita em mudança de NWC
    dcf_perpetual_growth_rate = 0.025 # 2.5% crescimento perpétuo

    # Múltiplos
    results['P/E'] = calculate_pe(info)
    results['P/B'] = calculate_pb(info)
    results['P/S'] = calculate_ps(info)
    results['EV/EBITDA'] = calculate_ev_ebitda(info)
    results['EV/EBIT'] = calculate_ev_ebit(info, financials)
    results['EV/Receita'] = calculate_ev_revenue(info, financials)

    # DDM
    results['DDM Stable Growth'] = ddm_stable_growth(info, dividends, ddm_discount_rate, ddm_growth_stable)
    results['DDM Multi-Stage'] = ddm_multi_stage(info, dividends, ddm_discount_rate, ddm_growth_multi_stage, ddm_growth_periods)

    # DCF 5 anos
    results['DCF 5 anos - saída via receita'] = dcf_model(info, financials, cash_flow, 5, dcf_revenue_growth_rate_5y, dcf_ebitda_margin, dcf_capex_as_pct_revenue, dcf_nwc_as_pct_revenue, dcf_perpetual_growth_rate)
    # Para as saídas via EBITDA e Crescimento, o modelo DCF base já incorpora a projeção de FCF.
    # A 


    # Para as saídas via EBITDA e Crescimento, o modelo DCF base já incorpora a projeção de FCF.
    # A diferença estaria na forma como o FCF é projetado ou o TV é calculado.
    # No entanto, para simplificar e seguir a estrutura de 


    # A diferença estaria na forma como o FCF é projetado ou o TV é calculado.
    # No entanto, para simplificar e seguir a estrutura de solicitação, usaremos o mesmo modelo DCF
    # e apenas ajustaremos os parâmetros de crescimento ou margem se necessário para simular as 'saídas'.
    # Para este caso, o modelo `dcf_model` já é genérico o suficiente.
    results["DCF 5 anos - saída via EBITDA"] = dcf_model(info, financials, cash_flow, 5, dcf_revenue_growth_rate_5y, dcf_ebitda_margin, dcf_capex_as_pct_revenue, dcf_nwc_as_pct_revenue, dcf_perpetual_growth_rate)
    results["DCF 5 anos - saída via crescimento"] = dcf_model(info, financials, cash_flow, 5, dcf_revenue_growth_rate_5y, dcf_ebitda_margin, dcf_capex_as_pct_revenue, dcf_nwc_as_pct_revenue, dcf_perpetual_growth_rate)

    # DCF 10 anos
    results["DCF 10 anos - saída via receita"] = dcf_model(info, financials, cash_flow, 10, dcf_revenue_growth_rate_10y, dcf_ebitda_margin, dcf_capex_as_pct_revenue, dcf_nwc_as_pct_revenue, dcf_perpetual_growth_rate)
    results["DCF 10 anos - saída via EBITDA"] = dcf_model(info, financials, cash_flow, 10, dcf_revenue_growth_rate_10y, dcf_ebitda_margin, dcf_capex_as_pct_revenue, dcf_nwc_as_pct_revenue, dcf_perpetual_growth_rate)
    results["DCF 10 anos - saída via crescimento"] = dcf_model(info, financials, cash_flow, 10, dcf_revenue_growth_rate_10y, dcf_ebitda_margin, dcf_capex_as_pct_revenue, dcf_nwc_as_pct_revenue, dcf_perpetual_growth_rate)

    # Earnings Power Value
    results["Earnings Power Value"] = earnings_power_value(info, financials, balance_sheet, ddm_discount_rate) # Reutilizando ddm_discount_rate como taxa de desconto

    return results


