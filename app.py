# app.py
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# --- 1. DATA PREPARATION ---
st.title("Explainable Credit Intelligence 💡")

# Load the data we saved earlier
stock_prices = pd.read_csv('stock_prices.csv', header=[0, 1], index_col=0, parse_dates=True)
news = pd.read_csv('news_headlines.csv', parse_dates=['date'])

def prepare_data_for_model(ticker):
    """Combines stock and news data to create features for our model."""
    
    # Get stock data for the selected ticker
    df = stock_prices.xs(ticker, level=1, axis=1).copy()
    
    # Feature 1: 7-day price change
    df['price_change_7d'] = df['Close'].pct_change(periods=7)

    # Feature 2: A simple news sentiment proxy (count of negative-sounding words)
    negative_words = ['fail', 'drop', 'loss', 'poor', 'bad', 'decline']
    news['sentiment'] = news['headline'].str.lower().apply(lambda h: sum(word in h for word in negative_words))
    daily_sentiment = news[news['ticker'] == ticker].groupby('date')['sentiment'].sum()
    
    # Combine features
    df = df.join(daily_sentiment)
    df['sentiment'] = df['sentiment'].fillna(0) # Fill days with no news with 0
    
    # Target Variable: Did the price go down in the next 7 days? (1 for yes, 0 for no)
    df['target'] = (df['Close'].shift(-7) < df['Close']).astype(int)
    
    df = df.dropna()
    return df

# --- 2. MODEL TRAINING ---
def train_model(df):
    """Trains a simple model and returns it with the SHAP explainer."""
    features = ['price_change_7d', 'sentiment']
    X = df[features]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Create the SHAP explainer
    explainer = shap.TreeExplainer(model)
    return model, explainer, X_test # Return test data for explanation

# --- 3. THE APP INTERFACE ---
ticker_symbol = st.selectbox("Select a Company Ticker", ('AAPL', 'MSFT', 'GOOGL'))

if ticker_symbol:
    # Prepare data and train model for the selected company
    model_data = prepare_data_for_model(ticker_symbol)
    model, explainer, X_test = train_model(model_data)

    st.header(f"Analysis for {ticker_symbol}")

    # --- Historical Score Trend ---
    st.subheader("Historical Score Trend")
    # Predict on the entire dataset to get historical probabilities
    historical_probs = model.predict_proba(model_data[['price_change_7d', 'sentiment']])
    # Get probability of "Lower Risk" (class 0) and convert to score
    historical_scores = pd.Series(historical_probs[:, 0] * 100, index=model_data.index)
    st.line_chart(historical_scores)

    st.subheader("Current Score Analysis")

    # --- Current Score Analysis (your existing code) ---
    latest_data = model_data.iloc[[-1]]

    # Get the latest data point to predict
    latest_data = model_data.iloc[[-1]]
    latest_features = latest_data[['price_change_7d', 'sentiment']]
    
    # Make prediction
    prediction_prob = model.predict_proba(latest_features)[0][0] # Probability of "price will not go down"
    score = int(prediction_prob * 100)
    
    st.subheader(f"Creditworthiness Score: {score}/100")
    if score < 60:
        st.warning("Higher Risk Detected")
    else:
        st.success("Lower Risk Detected")

  # --- 4. EXPLAIN THE PREDICTION (MODERN SHAP METHOD) ---
st.subheader("Why this score? (Feature Contribution)")

# Use the explainer to get a SHAP Explanation object
# This is the modern, preferred way to use SHAP
explanation = explainer(latest_features)

# For a binary classification, the explanation object has two sets of values.
# We want the values for the "risk" class, which is class 1.
# We also want the explanation for our single prediction, which is at index 0.
shap_values_for_risk_class = explanation[:,:,1][0]

# Create the plot using the new, more stable plotting function
st.write("#### Feature Impact Plot")
fig, ax = plt.subplots()
shap.plots.bar(shap_values_for_risk_class, show=False)
st.pyplot(fig, use_container_width=True)

st.write("""
**How to read this chart:** Features are ranked by their impact. Features pushing the risk **higher** (red bars) increase the score's risk. Features pushing the risk **lower** (blue bars) decrease it. The length of the bar shows the magnitude of the feature's impact for this specific prediction.
""")