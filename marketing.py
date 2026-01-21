import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules

st.title("📊 Marketing Analytics Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("Churn_pred.csv") 
    df.drop(columns=['customerID'], errors='ignore', inplace=True)

    
    categorical_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

df = load_data()

#Customer Lifetime Value (CLV) 
st.sidebar.subheader("🛒 Customer Lifetime Value (CLV)")


df['CLV'] = df['MonthlyCharges'] * df['tenure'] * 0.75  

X_clv = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y_clv = df['CLV']
clv_model = LinearRegression()
clv_model.fit(X_clv, y_clv)

st.sidebar.markdown("### Predict CLV for a New Customer")
tenure = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=10, max_value=200, value=50)
total_charges = tenure * monthly_charges

clv_prediction = clv_model.predict([[tenure, monthly_charges, total_charges]])[0]
st.sidebar.write(f"💰 Estimated CLV: **${clv_prediction:.2f}**")

#Customer Segmentation ---
st.subheader("🎯 Customer Segmentation for Marketing")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Segment'] = kmeans.fit_predict(df[['tenure', 'MonthlyCharges', 'TotalCharges']])
df['Segment'] = df['Segment'].map({0: 'Low Spend', 1: 'Medium Spend', 2: 'High Spend'})

seg_counts = df['Segment'].value_counts()
fig, ax = plt.subplots()
ax.pie(seg_counts, labels=seg_counts.index, autopct='%1.1f%%', colors=['green', 'orange', 'red'])
st.pyplot(fig)

st.write("### Recommended Marketing Strategies:")
st.write("- **High Spend Customers**: Exclusive offers, VIP rewards 🎁")
st.write("- **Medium Spend Customers**: Targeted promotions, cross-sell 📢")
st.write("- **Low Spend Customers**: Engage with free trials or discounts 💰")

#Market Basket Analysis 
st.subheader("🔗 Market Basket Analysis (Cross-Sell & Upsell)")

basket_cols = ['StreamingMovies', 'StreamingTV', 'OnlineSecurity', 'TechSupport', 'Contract']

df_basket = df[basket_cols].copy()

df_basket = df_basket.applymap(lambda x: 1 if x > 0 else 0) 

frequent_itemsets = apriori(df_basket, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

st.write("### Frequently Bought Services Together:")
st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

st.write("**📌 Insights for Marketing:**")
st.write("- If a customer has **StreamingTV**, offer **StreamingMovies** at a discount.")
st.write("- If a customer has **OnlineSecurity**, recommend **TechSupport**.")

# A/B Testing Simulation
st.subheader("📊 A/B Testing Simulation")

st.write("Compare two marketing strategies: **Discount vs. Premium Support**")
group_size = 1000
conversion_rate_A = st.slider("Conversion Rate (Discount)", 0.1, 1.0, 0.3)
conversion_rate_B = st.slider("Conversion Rate (Premium Support)", 0.1, 1.0, 0.4)

conversions_A = np.random.binomial(group_size, conversion_rate_A)
conversions_B = np.random.binomial(group_size, conversion_rate_B)

fig, ax = plt.subplots()
ax.bar(["Discount Offer", "Premium Support"], [conversions_A, conversions_B], color=['blue', 'orange'])
ax.set_ylabel("Conversions")
st.pyplot(fig)

if conversions_B > conversions_A:
    st.success("✅ **Premium Support Strategy Wins!**")
else:
    st.warning("🚀 **Discount Offer Strategy Wins!**")

