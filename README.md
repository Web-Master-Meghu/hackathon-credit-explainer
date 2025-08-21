# hackathon-credit-explainer
System Architecture: Briefly explain the flow: get_data.py fetches data from APIs and saves it to CSVs. app.py (Streamlit) loads this data, trains a model, predicts, explains with SHAP, and displays the results.


Tech Stack: List the technologies (Python, Streamlit, Pandas, Scikit-learn, SHAP, etc.) and justify why you chose them (e.g., "We chose Streamlit for its ability to rapidly create an interactive data science dashboard." ).

Trade-offs: Explain decisions you made. For example: "For this hackathon, we chose to save data to CSV files for simplicity and speed of development. In a production system, we would use a proper database like PostgreSQL for better scalability and data integrity." 