import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- Config ---
st.set_page_config(page_title="🏠 House Price Prediction", layout="wide")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")

df = load_data()

# --- Header ---
st.title("🏠 House Prices Dashboard")
st.markdown("Use the filters on the left to explore house sale prices, visualize trends, and predict prices with ML models.")

# --- Sidebar ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/31/House-icon.png/480px-House-icon.png", width=80)
st.sidebar.title("🔍 Filter Options")

neighborhoods = st.sidebar.multiselect(
    "Select Neighborhood(s):", df['Neighborhood'].unique(), default=list(df['Neighborhood'].unique())
)

year_range = st.sidebar.slider(
    "Year Built Range:",
    int(df['YearBuilt'].min()),
    int(df['YearBuilt'].max()),
    (int(df['YearBuilt'].min()), int(df['YearBuilt'].max()))
)

house_styles = st.sidebar.multiselect(
    "Select House Style(s):",
    df['HouseStyle'].unique(),
    default=list(df['HouseStyle'].unique())
)

overall_cond_range = st.sidebar.slider(
    "Overall Condition Range:",
    int(df['OverallCond'].min()),
    int(df['OverallCond'].max()),
    (int(df['OverallCond'].min()), int(df['OverallCond'].max()))
)

# --- Filter Data ---
with st.spinner("Filtering data..."):
    filtered_df = df[
        (df['Neighborhood'].isin(neighborhoods)) &
        (df['YearBuilt'] >= year_range[0]) & (df['YearBuilt'] <= year_range[1]) &
        (df['HouseStyle'].isin(house_styles)) &
        (df['OverallCond'] >= overall_cond_range[0]) & (df['OverallCond'] <= overall_cond_range[1])
    ]

# --- Tabs Layout ---
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🧠 Predict Price", "📁 Data & Model"])

# ------------------------ TAB 1 ------------------------
with tab1:
    st.subheader("📊 Visual Insights")
    st.markdown(f"Showing **{filtered_df.shape[0]} houses** after applying filters.")

    col1, col2 = st.columns(2)

    # Avg price by neighborhood
    with col1:
        st.markdown("#### Average Sale Price by Neighborhood")
        avg_price = filtered_df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
        bar_chart = alt.Chart(avg_price.reset_index()).mark_bar().encode(
            x=alt.X("Neighborhood", sort='-y'),
            y="SalePrice",
            tooltip=["Neighborhood", alt.Tooltip("SalePrice", format="$,.0f")]
        ).properties(height=350)
        st.altair_chart(bar_chart, use_container_width=True)

    # Histogram
    with col2:
        st.markdown("#### Sale Price Distribution")
        hist = alt.Chart(filtered_df).mark_bar().encode(
            alt.X("SalePrice", bin=alt.Bin(maxbins=30)),
            y='count()',
            tooltip=[alt.Tooltip('count()', title='Count')]
        )
        st.altair_chart(hist, use_container_width=True)

    st.markdown("#### Living Area vs Sale Price")
    scatter = alt.Chart(filtered_df).mark_circle(size=60).encode(
        x="GrLivArea",
        y="SalePrice",
        color="Neighborhood",
        tooltip=["GrLivArea", "SalePrice", "Neighborhood"]
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)

    # Summary Metrics
    st.markdown("### 📈 Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Sale Price", f"${filtered_df['SalePrice'].mean():,.0f}")
    col2.metric("Median Sale Price", f"${filtered_df['SalePrice'].median():,.0f}")
    col3.metric("Number of Houses", filtered_df.shape[0])

    st.markdown("___")
    st.subheader("📋 Filtered Data Preview")
    st.dataframe(filtered_df.head(50), use_container_width=True)

    # Download Button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Data", csv, "filtered_data.csv", "text/csv")

# ------------------------ TAB 2 ------------------------
with tab2:
    st.header("🧠 Predict House Sale Price")

    st.markdown("Input the house features below to get a price prediction:")

    overall_qual = st.slider('Overall Quality (1-10)', 1, 10, 5)
    gr_liv_area = st.slider('Living Area (sq ft)', int(df['GrLivArea'].min()), int(df['GrLivArea'].max()), 1500)
    year_built = st.slider('Year Built', int(df['YearBuilt'].min()), int(df['YearBuilt'].max()), 1980)
    overall_cond = st.slider('Overall Condition (1-10)', 1, 10, 5)
    garage_cars = st.slider('Garage Cars', 0, 4, 1)

    input_features = np.array([[overall_qual, gr_liv_area, year_built, overall_cond, garage_cars]])

    # Train model
    features = ['OverallQual', 'GrLivArea', 'YearBuilt', 'OverallCond', 'GarageCars']
    X = df[features]
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted_price = model.predict(input_features)[0]

    st.success(f"💰 Predicted Sale Price: ${predicted_price:,.2f}")

# ------------------------ TAB 3 ------------------------
with tab3:
    st.header("📁 Model Comparison")

    model_lr = LinearRegression()
    model_rf = RandomForestRegressor(random_state=42)

    model_lr.fit(X_train, y_train)
    model_rf.fit(X_train, y_train)

    pred_lr = model_lr.predict(X_test)
    pred_rf = model_rf.predict(X_test)

    rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
    rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
    r2_lr = r2_score(y_test, pred_lr)
    r2_rf = r2_score(y_test, pred_rf)

    st.write("### 📊 Model Performance on Test Set")
    st.write(f"- **Linear Regression**: RMSE = ${rmse_lr:,.0f}, R² = {r2_lr:.3f}")
    st.write(f"- **Random Forest**: RMSE = ${rmse_rf:,.0f}, R² = {r2_rf:.3f}")

    if rmse_rf < rmse_lr:
        st.success("🎯 Random Forest performed better.")
    else:
        st.success("🎯 Linear Regression performed better.")

# --- Footer ---
st.markdown("___")
st.markdown("Built with ❤️ by tenglexin & leeseeping • Powered by Streamlit")
