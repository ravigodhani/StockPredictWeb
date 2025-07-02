from flask import Flask, render_template, request
import yfinance as yf
from prophet import Prophet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import feedparser
import datetime

app = Flask(__name__)

COMMODITY_SYMBOLS = {
    'Crude Oil': 'CL=F',
    'USD/INR': 'USDINR=X'
}

def fetch_headlines(stock_name):
    url = f"https://news.google.com/rss/search?q={stock_name}+stock&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    return [entry.title for entry in feed.entries[:5]]

def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines if h]
    return round(sum(scores)/len(scores), 3) if scores else 0

def prepare_data(symbol):
    try:
        df = yf.download(symbol, start="2022-01-01", auto_adjust=True, progress=False)

        if df is None or df.empty:
            print(f"❌ DataFrame is empty for {symbol}")
            return None

        df = df.reset_index()
        date_col = 'Date' if 'Date' in df.columns else df.columns[0]
        close_col = 'Close' if 'Close' in df.columns else None

        if not close_col or date_col not in df.columns:
            print(f"❌ Required columns not found")
            return None

        df = df[[date_col, close_col]].copy()
        df.columns = ['ds', 'y']

        if not isinstance(df['ds'], pd.Series) or not isinstance(df['y'], pd.Series):
            print("❌ Columns are not Series type.")
            return None

        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df.dropna(inplace=True)

        if df.empty:
            print(f"❌ All data became NaN after cleanup for {symbol}")
            return None

        return df

    except Exception as e:
        print(f"❌ Exception while preparing data for {symbol}: {e}")
        return None

def predict_today(df):
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)
    today = pd.to_datetime(datetime.datetime.now().date())
    forecast_today = forecast[forecast['ds'] == today]
    if forecast_today.empty:
        forecast_today = forecast.tail(1)
    return forecast_today.iloc[0]

def get_commodity_prices():
    indicators = {}
    for name, symbol in COMMODITY_SYMBOLS.items():
        try:
            data = yf.download(symbol, period="1d", interval="1h", progress=False)
            indicators[name] = round(data['Close'][-1], 2)
        except:
            indicators[name] = "N/A"
    return indicators

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    if request.method == 'POST':
        stock = request.form.get('stock').upper().strip()
        symbol = f"{stock}.NS"
        df = prepare_data(symbol)
        if df is None or df.empty:
            error = f"Invalid or insufficient data for {stock}."
        else:
            try:
                forecast = predict_today(df)
                headlines = fetch_headlines(stock)
                sentiment = analyze_sentiment(headlines)
                indicators = get_commodity_prices()

                adjusted_yhat = round(forecast['yhat'] * (1 + sentiment * 0.01), 2)

                ticker = yf.Ticker(symbol)
                info = ticker.info
                stock_details = {
                    'name': info.get('longName'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'dividend_yield': info.get('dividendYield'),
                    'high_52': info.get('fiftyTwoWeekHigh'),
                    'low_52': info.get('fiftyTwoWeekLow'),
                    'day_high': info.get('dayHigh'),
                    'day_low': info.get('dayLow'),
                    'open': info.get('open'),
                    'prev_close': info.get('previousClose'),
                    'beta': info.get('beta'),
                    'description': info.get('longBusinessSummary')
                }

                result = {
                    'stock': stock,
                    'date': forecast['ds'].date(),
                    'yhat': round(forecast['yhat'], 2),
                    'yhat_lower': round(forecast['yhat_lower'], 2),
                    'yhat_upper': round(forecast['yhat_upper'], 2),
                    'adjusted_yhat': adjusted_yhat,
                    'sentiment': sentiment,
                    'headlines': headlines,
                    'indicators': indicators,
                    'details': stock_details
                }
            except Exception as e:
                error = str(e)
    return render_template('index.html', result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)
