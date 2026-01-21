import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Marketing Analytics Dashboard", page_icon="🚀", layout="wide")

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 28, 28, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Break long values in metrics if needed */
    div[data-testid="metric-container"] > div {
        overflow-wrap: break-word;
        white-space: pre-wrap;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Advanced Marketing Analytics Dashboard")

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("Churn_pred.csv")
        except FileNotFoundError:
            return None
    
    # Store original data for reference
    df_original = df.copy()
    
    # Remove customer ID if exists
    df.drop(columns=['customerID'], errors='ignore', inplace=True)
    
    # Handle missing values
    df = df.dropna()
    
    # Convert TotalCharges to numeric (often stored as string)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna(subset=['TotalCharges'])
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include='object').columns
    df_encoded = df.copy()
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Keep original categorical values for interpretation
    df_with_originals = df.copy()
    
    # Scale numerical variables for clustering
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df_scaled = df_encoded.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    return df_encoded, df_scaled, df_with_originals, scaler, label_encoders

# Sidebar for Data and Navigation
st.sidebar.title("🛠️ Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

# Load data
data_result = load_data(uploaded_file)

if data_result is None:
    st.warning("⚠️ Please upload a CSV file or ensure 'Churn_pred.csv' is in the directory.")
    st.stop()

df_encoded, df_scaled, df_original, scaler, label_encoders = data_result

st.sidebar.title("📋 Navigation")
analysis_type = st.sidebar.radio(
    "Choose Analysis:",
    ["Dashboard Overview", "Churn Prediction", "Customer Lifetime Value", "Customer Segmentation", "Market Basket Analysis", "A/B Testing"]
)

# --------------------------
# Dashboard Overview
# --------------------------
if analysis_type == "Dashboard Overview":
    st.header("📋 Executive Summary")
    
    # Ensure CLV is calculated for the overview (it might be missing if CLV section wasn't run)
    if 'CLV' not in df_encoded.columns:
         # Simple CLV approximation for overview
         if 'Churn' in df_encoded.columns:
             # If Churn is available (0/1), use it
             df_encoded['CLV'] = (df_encoded['MonthlyCharges'] * df_encoded['tenure'] * (1 - df_encoded['Churn'] * 0.5))
         else:
             df_encoded['CLV'] = (df_encoded['MonthlyCharges'] * df_encoded['tenure'])
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(df_encoded)
        st.metric("👥 Total Customers", f"{total_customers:,}")
    
    with col2:
        avg_clv = df_encoded['CLV'].mean()
        st.metric("💰 Avg Lifetime Value", f"${avg_clv:.2f}")
    
    with col3:
        avg_tenure = df_encoded['tenure'].mean()
        st.metric("📅 Avg Tenure", f"{avg_tenure:.1f} months")
    
    with col4:
        avg_monthly = df_encoded['MonthlyCharges'].mean()
        st.metric("💳 Avg Monthly", f"${avg_monthly:.2f}")
    
    st.markdown("---")
    
    # Overview Charts
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("📊 Customer Tenure Distribution")
        fig_tenure = px.histogram(df_encoded, x='tenure', nbins=30, title="Tenure Distribution", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig_tenure, use_container_width=True)
        
    with col_chart2:
        st.subheader("💳 Monthly Charges Distribution")
        fig_charges = px.histogram(df_encoded, x='MonthlyCharges', nbins=30, title="Monthly Charges Distribution", color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig_charges, use_container_width=True)
    
    if 'Churn' in df_encoded.columns:
        st.subheader("📉 Churn Rate Overview")
        # Handle if Churn is numeric (0/1) or categorical
        churn_counts = df_encoded['Churn'].value_counts()
        fig_churn = px.pie(values=churn_counts.values, names=churn_counts.index, title="Overall Churn Rate", 
                           color_discrete_sequence=px.colors.qualitative.Set3, hole=0.4)
        st.plotly_chart(fig_churn, use_container_width=True)

    st.success("💡 **Tip:** Use the sidebar to navigate to specific modules like Churn Prediction or Segmentation for deeper analysis.")

# --------------------------
# Churn Prediction (NEW)
# --------------------------
if analysis_type == "Churn Prediction":
    st.header("🔮 Churn Prediction Model")
    
    if 'Churn' not in df_encoded.columns:
        st.error("⚠️ 'Churn' column not found in dataset. Cannot perform churn prediction.")
    else:
        # Prepare data
        X = df_encoded.drop(columns=['Churn', 'CLV', 'Segment', 'Segment_Name'], errors='ignore')
        y = df_encoded['Churn']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎯 Model Performance")
            st.metric("Accuracy", f"{acc:.2%}")
            st.write("**Classification Report:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
            
        with col2:
            st.subheader("🔑 Feature Importance")
            feature_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
            fig = px.bar(feature_imp, x=feature_imp.values, y=feature_imp.index, orientation='h', 
                         title="Top 10 Drivers of Churn", labels={'x': 'Importance', 'index': 'Feature'},
                         color=feature_imp.values, color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Customer Lifetime Value (CLV)
# --------------------------
if analysis_type == "Customer Lifetime Value":
    st.header("💰 Customer Lifetime Value (CLV) Analysis")
    
    # Improved CLV calculation
    df_encoded['CLV'] = (df_encoded['MonthlyCharges'] * df_encoded['tenure'] * 
                        (1 - df_encoded.get('Churn', 0) * 0.5))  # Adjust for churn risk
    
    # Build CLV prediction model
    X_clv = df_encoded[['tenure', 'MonthlyCharges', 'TotalCharges']]
    y_clv = df_encoded['CLV']
    
    clv_model = LinearRegression()
    clv_model.fit(X_clv, y_clv)
    clv_r2 = r2_score(y_clv, clv_model.predict(X_clv))
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 CLV Predictor")
        st.write(f"**Model Accuracy (R²): {clv_r2:.3f}**")
        
        # Input sliders with realistic ranges
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=65.0, step=1.0)
        total_charges = st.slider("Total Charges ($)", min_value=18.0, max_value=8500.0, value=float(tenure * monthly_charges), step=10.0)
        
        # Predict CLV
        input_data = pd.DataFrame([[tenure, monthly_charges, total_charges]],
                                  columns=['tenure', 'MonthlyCharges', 'TotalCharges'])
        clv_prediction = clv_model.predict(input_data)[0]
        
        st.metric(label="💰 Estimated CLV", value=f"${clv_prediction:.2f}")
        
        # CLV interpretation
        if clv_prediction > df_encoded['CLV'].quantile(0.75):
            st.success("🌟 High Value Customer")
        elif clv_prediction > df_encoded['CLV'].quantile(0.25):
            st.info("📈 Medium Value Customer")
        else:
            st.warning("⚠️ Low Value Customer")
    
    with col2:
        st.subheader("📊 CLV Distribution")
        fig = px.histogram(df_encoded, x='CLV', nbins=50, title="Customer Lifetime Value Distribution",
                           color_discrete_sequence=['skyblue'], opacity=0.7)
        fig.add_vline(x=clv_prediction, line_dash="dash", line_color="red", 
                      annotation_text=f'Predicted: ${clv_prediction:.2f}', annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Customer Segmentation
# --------------------------
if analysis_type == "Customer Segmentation":
    st.header("🎯 Customer Segmentation Analysis")
    
    # Perform clustering
    features_for_clustering = ['tenure', 'MonthlyCharges', 'TotalCharges']
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_encoded['Segment'] = kmeans.fit_predict(df_scaled[features_for_clustering])
    
    # Create meaningful segment labels
    segment_profiles = df_encoded.groupby('Segment')[features_for_clustering].mean()
    
    segment_names = {}
    for i in range(4):
        tenure_avg = segment_profiles.loc[i, 'tenure']
        charges_avg = segment_profiles.loc[i, 'MonthlyCharges']
        if tenure_avg > df_encoded['tenure'].mean() and charges_avg > df_encoded['MonthlyCharges'].mean():
            segment_names[i] = 'Loyal High-Spenders'
        elif tenure_avg > df_encoded['tenure'].mean():
            segment_names[i] = 'Loyal Budget-Conscious'
        elif charges_avg > df_encoded['MonthlyCharges'].mean():
            segment_names[i] = 'New High-Spenders'
        else:
            segment_names[i] = 'New Budget-Conscious'
    
    df_encoded['Segment_Name'] = df_encoded['Segment'].map(segment_names)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 Customer Segments")
        seg_counts = df_encoded['Segment_Name'].value_counts()
        fig = px.pie(names=seg_counts.index, values=seg_counts.values, title='Customer Segment Distribution',
                     hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Segment Characteristics")
        segment_details = df_encoded.groupby('Segment_Name').agg({
            'tenure': 'mean',
            'MonthlyCharges': 'mean',
            'TotalCharges': 'mean',
            'Segment': 'count'
        }).round(2)
        segment_details.columns = ['Avg Tenure', 'Avg Monthly Charges', 'Avg Total Charges', 'Count']
        st.dataframe(segment_details)
    
    st.subheader("💡 Recommended Marketing Strategies")
    strategies = {
        'Loyal High-Spenders': "🌟 VIP treatment, exclusive offers, premium support, loyalty rewards",
        'Loyal Budget-Conscious': "🎁 Volume discounts, referral bonuses, appreciation programs",
        'New High-Spenders': "🚀 Premium service upgrades, early access to new features",
        'New Budget-Conscious': "💰 Welcome discounts, free trials, gradual upselling"
    }
    
    for segment, count in seg_counts.items():
         if segment in strategies:
            st.info(f"**{segment}** ({count} customers): {strategies[segment]}")

# --------------------------
# Market Basket Analysis
# --------------------------
if analysis_type == "Market Basket Analysis":
    st.header("🛒 Market Basket Analysis (Cross-Sell Opportunities)")
    
    # Select boolean/binary service columns
    service_cols = [col for col in df_original.columns if df_original[col].nunique() == 2 and col != 'Churn']
    
    if len(service_cols) > 0:
        st.subheader("📊 Service Adoption Analysis")
        
        # Convert to boolean for market basket analysis
        df_basket = df_original[service_cols].copy()
        
        # Handle different encoding formats (Yes/No, 1/0)
        for col in service_cols:
            unique_vals = df_basket[col].unique()
            if 'Yes' in unique_vals:
                df_basket[col] = (df_basket[col] == 'Yes')
            elif 'No' in unique_vals:
                df_basket[col] = (df_basket[col] != 'No')
            else:
                df_basket[col] = df_basket[col].astype(bool)
        
        # Generate frequent itemsets
        try:
            frequent_itemsets = apriori(df_basket, min_support=0.1, use_colnames=True)
            
            if len(frequent_itemsets) > 0:
                # Generate association rules
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.1)
                
                if len(rules) > 0:
                    # Convert frozensets to strings
                    rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    
                    # Display top rules
                    st.subheader("🔗 Top Cross-Sell Opportunities")
                    top_rules = rules.nlargest(10, 'lift')[['antecedents_str', 'consequents_str', 'support', 'confidence', 'lift']]
                    top_rules.columns = ['If Customer Has', 'Recommend', 'Support', 'Confidence', 'Lift']
                    st.dataframe(top_rules.round(3))
                    
                    # Service adoption heatmap
                    st.subheader("🔥 Service Adoption Heatmap")
                    corr_matrix = df_basket.corr()
                    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                                    title="Service Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Marketing insights
                    st.subheader("💡 Key Marketing Insights")
                    if len(rules) > 0:
                        best_rule = rules.loc[rules['lift'].idxmax()]
                        st.success(f"🎯 **Best Cross-sell Opportunity**: If customer has {list(best_rule['antecedents'])[0]}, recommend {list(best_rule['consequents'])[0]} (Lift: {best_rule['lift']:.2f})")
                        
                        st.write("**Recommended Actions:**")
                        for _, rule in rules.head(3).iterrows():
                            antecedent = list(rule['antecedents'])[0]
                            consequent = list(rule['consequents'])[0]
                            st.write(f"• Target customers with **{antecedent}** for **{consequent}** promotions")
                else:
                    st.warning("No significant association rules found. Try lowering the minimum threshold.")
            else:
                st.warning("No frequent itemsets found. The data might be too sparse.")
        except Exception as e:
            st.error(f"Error in market basket analysis: {str(e)}")
    else:
        st.warning("No suitable service columns found for market basket analysis.")

# --------------------------
# A/B Testing Simulation
# --------------------------
if analysis_type == "A/B Testing":
    st.header("🧪 A/B Testing Simulation")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Test Configuration")
        
        # Test parameters
        group_size = st.number_input("Sample Size per Group", min_value=100, max_value=10000, value=1000, step=100)
        
        strategy_A = st.text_input("Strategy A Name", value="Discount Offer")
        conversion_rate_A = st.slider("Strategy A Conversion Rate", 0.01, 0.50, 0.15, 0.01)
        
        strategy_B = st.text_input("Strategy B Name", value="Premium Support")
        conversion_rate_B = st.slider("Strategy B Conversion Rate", 0.01, 0.50, 0.20, 0.01)
        
        # Run simulation
        np.random.seed(42)  # For reproducible results
        conversions_A = np.random.binomial(group_size, conversion_rate_A)
        conversions_B = np.random.binomial(group_size, conversion_rate_B)
        
        # Statistical significance test (simplified)
        difference = abs(conversions_B - conversions_A)
        relative_improvement = (conversions_B - conversions_A) / conversions_A * 100 if conversions_A > 0 else 0
    
    with col2:
        st.subheader("📈 Results")
        
        # Display metrics
        col2_1, col2_2 = st.columns(2)
        
        with col2_1:
            st.metric(label=strategy_A, value=f"{conversions_A}", delta=f"{conversions_A/group_size*100:.1f}% conversion")
        
        with col2_2:
            st.metric(label=strategy_B, value=f"{conversions_B}", delta=f"{conversions_B/group_size*100:.1f}% conversion")
        
        # Visualization
        strategies = [strategy_A, strategy_B]
        conversions = [conversions_A, conversions_B]
        
        fig = px.bar(x=strategies, y=conversions, color=strategies, title="A/B Test Results",
                     labels={'x': 'Strategy', 'y': 'Conversions'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical conclusion
        if conversions_B > conversions_A:
            st.success(f"✅ **{strategy_B} Wins!** (+{relative_improvement:.1f}% improvement)")
        elif conversions_A > conversions_B:
            st.success(f"✅ **{strategy_A} Wins!** (+{abs(relative_improvement):.1f}% better)")
        else:
            st.info("🤝 **It's a Tie!** Both strategies performed equally")
        
        # Confidence interval (simplified)
        st.write("**📊 Statistical Summary:**")
        st.write(f"• Sample size: {group_size:,} per group")
        st.write(f"• Absolute difference: {difference} conversions")
        st.write(f"• Relative improvement: {relative_improvement:.1f}%")