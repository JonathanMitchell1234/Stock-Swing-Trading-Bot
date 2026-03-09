import requests

# Insert your Alpaca Paper or Live keys here
api_key = "PKEZW3OSHW6LUBJDHCECXHW5E6"
secret_key = "85C7BwCuvmqffRxR4iWuRrR71Ni14wB4rGgPrqduWgNT"
base_url = "https://paper-api.alpaca.markets" # Change to api.alpaca.markets for live

headers = {
    "APCA-API-KEY-ID": api_key,
    "APCA-API-SECRET-KEY": secret_key
}

tickers = [
    "AIT", "AME", "BDC", "CGNX", "EMR", "GD", "GXO", "HON", 
    "ISRG", "JNJ", "MDT", "NOVT", "OMCL", "ROK", "RRX", "SYK", "TER"
]

for ticker in tickers:
    url = f"{base_url}/v2/assets/{ticker}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        tradable = data.get("tradable")
        fractionable = data.get("fractionable")
        
        # We only care if Alpaca has flagged it as unsupported
        if not tradable or not fractionable:
            print(f"Flagged -> {ticker} | Tradable: {tradable} | Fractionable: {fractionable}")
    else:
        print(f"Error fetching {ticker}: {response.status_code}")