import logging
from broker import AlpacaBroker
from sentiment import get_sentiment

# Set up basic logging to see the output
logging.basicConfig(level=logging.INFO, format="%(message)s")

def run_test():
    broker = AlpacaBroker()
    
    print("\n--- TEST 1: LIVE ALPACA NEWS (Over the Weekend) ---")
    tickers_to_test = ["TTD", "ETON", "MGNI"]
    
    for symbol in tickers_to_test:
        headlines = broker.get_news(symbol, limit=5)
        print(f"\nNews for {symbol}:")
        for h in headlines:
            print(f"- {h}")
            
        score = get_sentiment(headlines)
        print(f"--> Aggregated Sentiment Score: {score:.3f}")
        if score < -0.20:
            print("--> ACTION: Screener would REJECT this trade.")
        else:
            print("--> ACTION: Screener would ALLOW this trade.")

    print("\n\n--- TEST 2: FAKE EXTREME HEADLINES ---")
    
    bad_news = [
        "The company unexpectedly filed for bankruptcy today.",
        "CEO steps down amidst massive accounting fraud scandal.",
        "Earnings miss expectations by a huge margin, stock plummets."
    ]
    bad_score = get_sentiment(bad_news)
    print("\nFake Bad News:")
    for h in bad_news: print(f"- {h}")
    print(f"--> Aggregated Sentiment Score: {bad_score:.3f}")

    good_news = [
        "Company smashes earnings expectations, reporting record profits.",
        "Analyst upgrades stock to strong buy with massive price target increase.",
        "Revenue grew by 500% year over year."
    ]
    good_score = get_sentiment(good_news)
    print("\nFake Good News:")
    for h in good_news: print(f"- {h}")
    print(f"--> Aggregated Sentiment Score: {good_score:.3f}")

if __name__ == "__main__":
    run_test()
