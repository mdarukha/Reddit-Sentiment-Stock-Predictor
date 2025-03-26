# 📈 Reddit Sentiment Stock Predictor: Predicting Stock Prices from Reddit Sentiment and Financial Data

A full end-to-end machine learning pipeline that predicts stock prices using sentiment analysis on Reddit posts combined with traditional market data. This project fine-tunes a financial BERT model, scrapes real-world Reddit discussions, engineers market features, and applies advanced machine learning models to forecast stock prices for any public company.

---

## 💡 What This Project Does

1. **Fine-tunes FinBERT** (a BERT model pre-trained on financial text) using labeled Twitter financial sentiment data.
2. **Scrapes Reddit posts** from general stock subreddits and company-specific subs (like `r/wallstreetbets`, `r/NVDA_Stock`, or `r/AAPL`).
3. **Performs sentiment analysis** on Reddit posts using the fine-tuned model to produce daily sentiment scores.
4. **Scrapes historical stock data** from Yahoo Finance and engineers features:
   - Lag features (1-day, 2-day, 3-day)
   - Moving averages (5-day, 10-day)
   - Daily return and log return
5. **Trains multiple models** to predict future stock prices:
   - Random Forest
   - XGBoost
   - LSTM
   - RNN
6. **Evaluates performance** using MAE, MSE, and R²
7. **Visualizes predictions and errors** with Matplotlib and Seaborn

---

## 📁 Project Structure

```
├── data/
│   ├── posts.csv           # Historical Reddit post data
│   ├── stock_index.csv     # Historical Yahoo Finance data
│   ├── appl_sentiment_analysis.csv     # Sentiment scores for scraped AAPL subreddit posts
│   ├── aapl_stock_sentiment_features.csv     # Engineered features for AAPL used in models
├── main_notebook.ipynb       # Main notebook with pipeline
├── README.md
```

posts.csv and stock_index.csv datasets obtained from: https://www.kaggle.com/datasets/injek0626/reddit-stock-related-posts?select=stock_index.csv


---

## 🛠️ Tools & Libraries

- `transformers` by HuggingFace for FinBERT fine-tuning
- `PRAW` for Reddit scraping
- `yfinance` for historical price data
- `scikit-learn` for traditional ML models
- `XGBoost` for gradient boosting
- `TensorFlow / Keras` for deep learning (LSTM + RNN)
- `Matplotlib`, `Seaborn` for visualizations

---

## 🧠 Models & Results

| Model         | MAE ($)  | R² Score |
|---------------|----------|---------:|
| Random Forest |   ~2.00  |   ~0.92  |
| XGBoost       |   ~1.85  |   ~0.94  |
| LSTM          |   ~2.75  |   ~0.88  |
| RNN           |   ~2.90  |   ~0.86  |

> ✅ MAE and RMSE are in **dollars**, providing intuitive interpretability for price prediction.

---

## 🔄 How to Predict for Any Stock Ticker

You can easily adapt this pipeline to any stock:

### ✅ Step 1: Set up API credentials

You will need Reddit API credentials to run the notebook:

- Go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
- Create a new "script" app
- Use your `client_id`, `client_secret`, and `user_agent` in the PRAW setup

```python
reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    user_agent='Stock Sentiment Analysis Script'
)
```

### ✅ Step 2: Update the ticker and subreddit list

```python
# Example for Tesla
ticker = "TSLA"
subreddits = ["TSLA", "teslainvestorsclub", "wallstreetbets", "stocks"]
```

### ✅ Step 3: Re-run the notebook
- Scraping will dynamically collect Reddit posts mentioning the company
- Yahoo Finance will automatically pull stock price history
- Model training, evaluation, and visualization will adapt to the new stock

---

## ✨ Visualizations (shown in notebook for AAPL and NVDA)

- 📈 Actual vs. predicted stock price plot
- 📉 Model error (absolute) over time
- 🔥 XGBoost feature importance heatmap
- 📊 Correlation matrix of engineered features

---

## 📉 Error Metrics

- **MAE (Mean Absolute Error)** represents average dollars off: e.g., MAE = 2.5 → predictions are on average $2.50 off.
- **RMSE (Root Mean Squared Error)** also in dollars but penalizes large deviations more.

Both are used to evaluate stock price prediction accuracy.

---

## 🚀 Future Work

- 🔄 Automate scraping and model retraining on a schedule using Airflow or cron jobs
- 📈 Deploy the model as a live dashboard or streamlit app
- 🧠 Fine-tune FinBERT using Reddit-specific financial posts for improved context accuracy
- 🔍 Incorporate other alternative data sources like Twitter or earnings call transcripts
- 🧮 Try forecasting next-day returns instead of price, or classifying bullish/bearish movements
- 🧠 Implement attention-based transformer models for time-series prediction
- 🌎 Expand to portfolio-level prediction using multiple tickers and cross-correlated sentiment
- 🧪 Add backtesting with trading strategies and evaluate profitability

---

