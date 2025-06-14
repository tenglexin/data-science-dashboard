import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Page settings
st.set_page_config(page_title="🏠 House Price Prediction", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")

df = load_data()

# ✅ Page Title (appears above everything, before tabs)
st.title("🏠 House Prices - Dashboard")
st.markdown("Use the filters on the left and explore price insights, predictions, and model performance.")

# Sidebar - Filters
st.sidebar.title("🔍 Filter Options")
neighborhoods = st.sidebar.multiselect("Neighborhood", df["Neighborhood"].unique(), default=list(df["Neighborhood"].unique()))
year_range = st.sidebar.slider("Year Built Range", int(df["YearBuilt"].min()), int(df["YearBuilt"].max()), (2000, 2010))
house_styles = st.sidebar.multiselect("House Style", df["HouseStyle"].unique(), default=list(df["HouseStyle"].unique()))
overall_cond = st.sidebar.slider("Overall Condition", int(df["OverallCond"].min()), int(df["OverallCond"].max()), (3, 9))

# Filter dataset
filtered_df = df[
    (df["Neighborhood"].isin(neighborhoods)) &
    (df["YearBuilt"] >= year_range[0]) &
    (df["YearBuilt"] <= year_range[1]) &
    (df["HouseStyle"].isin(house_styles)) &
    (df["OverallCond"] >= overall_cond[0]) &
    (df["OverallCond"] <= overall_cond[1])
]

# Tabs for layout
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Predict Price", "📁 Data & Model"])

with tab1:
    st.header("📊 Data Visualizations")
    st.markdown(f"Showing **{filtered_df.shape[0]} houses** after applying filters.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Sale Price by Neighborhood")
        avg_price_by_neigh = filtered_df.groupby("Neighborhood")["SalePrice"].mean().sort_values(ascending=False)
        bar_chart = alt.Chart(avg_price_by_neigh.reset_index()).mark_bar().encode(
            x=alt.X("Neighborhood", sort="-y"),
            y="SalePrice",
            tooltip=["Neighborhood", alt.Tooltip("SalePrice", format="$,.0f")]
        ).properties(height=400)
        st.altair_chart(bar_chart, use_container_width=True)

    with col2:
        st.subheader("Sale Price Distribution")
        hist = alt.Chart(filtered_df).mark_bar().encode(
            x=alt.X("SalePrice", bin=alt.Bin(maxbins=30)),
            y="count()",
            tooltip=["count()", "SalePrice"]
        ).properties(height=400)
        st.altair_chart(hist, use_container_width=True)

    st.subheader("Living Area vs. Sale Price")
    scatter = alt.Chart(filtered_df).mark_circle(size=60).encode(
        x="GrLivArea",
        y="SalePrice",
        color="Neighborhood",
        tooltip=["Neighborhood", "GrLivArea", "SalePrice"]
    ).interactive()
    st.altair_chart(scatter, use_container_width=True)

    with st.expander("📥 Download Filtered Data"):
        st.download_button(
            label="Download CSV",
            data=filtered_df.to_csv(index=False),
            file_name="filtered_house_data.csv",
            mime="text/csv"
        )

with tab2:
    st.header("💡 Predict House Sale Price")

    st.markdown("Use the sliders below to input house features and get a predicted sale price:")

    col1, col2, col3 = st.columns(3)

    with col1:
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
        overall_cond_input = st.slider("Overall Condition (1-10)", 1, 10, 5)

    with col2:
        gr_liv_area = st.slider("Living Area (sq ft)", 300, 6000, 1500)
        garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 1)

    with col3:
        year_built = st.slider("Year Built", 1872, 2010, 1995)

    # Prediction
    input_data = np.array([[overall_qual, gr_liv_area, year_built, overall_cond_input, garage_cars]])

    features = ['OverallQual', 'GrLivArea', 'YearBuilt', 'OverallCond', 'GarageCars']
    X = df[features]
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predicted_price = model.predict(input_data)[0]

    st.success(f"💰 Predicted Sale Price: **${predicted_price:,.2f}**")

    with st.expander("📉 Model Performance on Test Set"):
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.metric("RMSE", f"${rmse:,.0f}")
        st.metric("R² Score", f"{r2:.3f}")

with tab3:
    st.header("📁 Data Summary and Model Info")

    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Price", f"${filtered_df['SalePrice'].mean():,.0f}")
    col2.metric("Median Price", f"${filtered_df['SalePrice'].median():,.0f}")
    col3.metric("Listings Displayed", filtered_df.shape[0])

    st.subheader("🧠 Feature Importance (Linear Coefficients)")
    coefs = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", ascending=False)

    coef_chart = alt.Chart(coefs).mark_bar().encode(
        x=alt.X("Coefficient"),
        y=alt.Y("Feature", sort="-x")
    ).properties(height=300)

    st.altair_chart(coef_chart, use_container_width=True)

    st.write("All data and modeling are based on the Kaggle competition dataset.")

