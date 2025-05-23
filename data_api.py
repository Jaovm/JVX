# data_api.py
import requests

class ApiClient:
    def __init__(self):
        self.base_url = "https://api.apilayer.com/yahoo"
        self.headers = {
            "apikey": "ChCJBikflgPfGXfoYqmOLscLvmSGB8WN"  # Substitua pela sua chave da Apilayer
        }

    def call_api(self, endpoint, query=None):
        if endpoint == 'YahooFinance/get_stock_chart':
            symbol = query.get('symbol')
            interval = query.get('interval', '1mo')
            range_ = query.get('range', '5y')

            url = f"{self.base_url}/time-series"
            params = {
                "symbol": symbol,
                "interval": interval,
                "range": range_,
                "region": query.get('region', 'BR')
            }

        elif endpoint == 'YahooFinance/get_stock_insights':
            symbol = query.get('symbol')
            url = f"{self.base_url}/insights"
            params = {"symbol": symbol}

        else:
            print(f"Endpoint '{endpoint}' não suportado.")
            return None

        try:
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Erro {response.status_code} na chamada da API: {response.text}")
                return None
        except Exception as e:
            print(f"Erro na chamada da API: {e}")
            return None
