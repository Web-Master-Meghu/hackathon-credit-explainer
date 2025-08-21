# Explainable Credit Intelligence Platform ✨

🏆 **Submission for the CredTech Hackathon | Organized by The Programming Club, IITK**

[cite_start]This project tackles the challenge of slow and opaque credit ratings by providing a real-time, data-driven, and fully explainable creditworthiness score for public companies. [cite: 6, 7]

---

## 🚀 Live Demo

**[>> You can access the live application here <<](https://hackathon-credit-explainer-fgxt2dp2ayuwy42rb5me5j.streamlit.app/)**

## 💡 Key Features

* [cite_start]**Dynamic Credit Scoring:** Generates a daily creditworthiness score that reacts much faster to market events than traditional agency ratings. [cite: 19]
* [cite_start]**Multi-Source Data Ingestion:** Fuses data from multiple sources to get a holistic view of credit risk. [cite: 18]
    * [cite_start]**Structured Data:** Daily stock prices (Yahoo Finance) and macroeconomic indicators (FRED API). [cite: 27, 28]
    * [cite_start]**Unstructured Data:** Real-time news sentiment analysis from financial headlines (NewsAPI). [cite: 29]
* **AI-Powered Explainability:** We don't just give you a score; we show you *why*. [cite_start]Using SHAP (SHapley Additive exPlanations), our dashboard provides a clear, feature-level breakdown of what factors are driving the score, turning the "black box" into a transparent tool. [cite: 15, 20, 42]
* [cite_start]**Interactive Analyst Dashboard:** A clean, user-friendly web interface built with Streamlit that visualizes historical score trends and the latest feature contributions. [cite: 22, 48]

## 🏗️ System Architecture

Our architecture is designed for simplicity and rapid development, perfect for a hackathon. The data flows in a straightforward, two-step process:

1.  **Data Ingestion Script (`get_data.py`):** A Python script runs on a schedule to fetch the latest data from our chosen APIs (Yahoo Finance, NewsAPI, FRED). It processes this data and saves it into clean CSV files.
2.  **Streamlit Application (`app.py`):** The interactive web app loads the prepared data from the CSVs. When a user selects a company, it trains a machine learning model on the fly, calculates the current credit score, generates the SHAP explanation, and displays all the results.

## 🛠️ Tech Stack & Justification

* **Backend:** Python
* **Data Processing:** Pandas, Scikit-learn
* **APIs:** yfinance, newsapi-python, fredapi
* **Machine Learning:**
    * **Model:** RandomForestClassifier - A robust and effective model for tabular data that performs well without extensive tuning.
    * [cite_start]**Explainability:** SHAP - The core of our project, providing trustworthy, feature-level explanations without using LLMs. [cite: 45, 46]
* **Frontend & Deployment:**
    * **Dashboard:** Streamlit - Chosen for its incredible speed in turning data scripts into beautiful, interactive web applications.
    * [cite_start]**Containerization:** Docker - A `Dockerfile` is included to meet reproducibility requirements and demonstrate a path to production. [cite: 53]
    * **Hosting:** Streamlit Community Cloud - For easy and free deployment directly from our GitHub repository.

## ⚙️ How to Run Locally

To get this project running on your own machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone (https://github.com/Web-Master-Meghu/hackathon-credit-explainer.git)
    cd your-repo-name
    ```
2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Add your API keys:**
    * Open `get_data.py`.
    * Add your free API keys for NewsAPI and FRED where indicated.
5.  **Fetch the data:**
    * This is a crucial step! Run the data script first.
    ```bash
    python get_data.py
    ```
6.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    Your browser should open with the app running!

## ⚖️ Key Trade-offs & Future Improvements

* **Data Storage:** We chose to use **CSVs as our data store for simplicity and speed** during this hackathon. This was a trade-off against using a more robust solution like a PostgreSQL database. [cite_start]In a production environment, a database would be essential for handling high-volume data, ensuring data integrity, and allowing for more complex queries. [cite: 74]
* **Model Retraining:** Our model is currently retrained on each app load for the selected company. [cite_start]A more advanced MLOps approach would involve a scheduled, automated pipeline (e.g., using Airflow) to retrain the model on new data and version it, which was beyond the scope of this hackathon but is a clear next step. [cite: 54]

## 📹 Video Walkthrough

**[>> Link to our 5-7 minute video demonstration <<](YOUR_YOUTUBE_OR_LOOM_LINK)**
