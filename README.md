# 🚀 Advanced Marketing Analytics Dashboard

A powerful, interactive Streamlit application designed to help marketing teams analyze customer behavior, predict churn, and optimize strategies using Machine Learning.

## 📌 Problem Statement
In the competitive landscape of subscription-based businesses (like Telecom, SaaS, or Media), retaining customers is often more cost-effective than acquiring new ones. Marketing teams struggle with:
- **Identifying who will leave (Churn):** effectively intervening before a customer cancels.
- **Understanding Customer Value:** Knowing which customers are "VIPs" versus low-value.
- **Segmentation:** Grouping customers by behavior rather than just demographics.
- **Cross-Selling:** Knowing what additional services to recommend.
- **Testing Strategies:** Simulating the outcome of marketing campaigns before spending money.

## 💡 The Solution
This project provides a **unified data science dashboard** that solves these problems using:
1.  **🔮 Churn Prediction:** Uses a Random Forest Classifier to identify at-risk customers with ~80% accuracy and explains *why* they might leave (Feature Importance).
2.  **💰 Customer Lifetime Value (CLV):** Predicts the long-term value of a customer to help prioritize retention efforts.
3.  **🎯 Smart Segmentation:** Uses K-Means Clustering to group customers into 4 actionable profiles (e.g., "Loyal High-Spenders", "New Budget-Conscious").
4.  **🛒 Market Basket Analysis:** Uses Association Rule Mining (Apriori) to find hidden patterns in service adoption (e.g., "People with Streaming TV often buy Movies").
5.  **🧪 A/B Testing Simulator:** A statistical tool to simulate and visualize the potential outcome of two different marketing strategies.

## ✨ Features
- **Zero-Setup Model Training:** data is trained on-the-fly, so no need to manage complex `.pkl` (pickle) files.
- **Interactive Visualizations:** Powered by Plotly for zooming, panning, and hovering.
- **Dynamic Data Loading:** Upload your own CSV files to analyze different datasets instantly.
- **Dark Mode UI:** Professional, high-contrast interface.

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd marketing-analytics-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run churn.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

## 📂 Project Structure
- `churn.py`: The main application code containing all logic and UI.
- `Churn_pred.csv`: Default dataset for demonstration.
- `requirements.txt`: List of Python libraries required.
- `README.md`: Project documentation.

## 📊 Technologies Used
- **Streamlit:** For the web interface.
- **Pandas & NumPy:** For data manipulation.
- **Scikit-Learn:** For Machine Learning (Random Forest, K-Means, Linear Regression).
- **Plotly:** For interactive charting.
- **Mlxtend:** For Market Basket Analysis.

---
*Built for the Marketing Analytics Portfolio.*
