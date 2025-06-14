import streamlit as st
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv('data/train.csv')

st.title("🏠 House Prices Dashboard")

# --- Sidebar for filters and prediction inputs ---
with st.sidebar:
    st.header("Filters")

    neighborhoods = st.multiselect(
        'Select Neighborhood(s):',
        options=df['Neighborhood'].unique(),
        default=df['Neighborhood'].unique()
    )

    min_year = int(df['YearBuilt'].min())
    max_year = int(df['YearBuilt'].max())
    year_range = st.slider(
        'Select Year Built Range:',
        min_year,
        max_year,
        (min_year, max_year)
    )

    house_styles = st.multiselect(
        "Select House Style(s):",
        options=df['HouseStyle'].unique(),
        default=df['HouseStyle'].unique()
    )

    overall_cond_range = st.slider(
        "Select Overall Condition Range:",
        int(df['OverallCond'].min()),
        int(df['OverallCond'].max()),
        (int(df['OverallCond'].min()), int(df['OverallCond'].max()))
    )

    st.header("Prediction Inputs")

    overall_qual = st.slider('Overall Quality (1-10)', int(df['OverallQual'].min()), int(df['OverallQual'].max()), 5)
    gr_liv_area = st.slider('Living Area (sq ft)', int(df['GrLivArea'].min()), int(df['GrLivArea'].max()), 1500)
    year_built = st.slider('Year Built', int(df['YearBuilt'].min()), int(df['YearBuilt'].max()), 1980)
    overall_cond = st.slider('Overall Condition (1-10)', int(df['OverallCond'].min()), int(df['OverallCond'].max()), 5)
    garage_cars = st.slider('Garage Cars', int(df['GarageCars'].min()), int(df['GarageCars'].max()), 1)

# --- Filter the dataframe ---
filtered_df = df[
    (df['Neighborhood'].isin(neighborhoods)) &
    (df['YearBuilt'] >= year_range[0]) & (df['YearBuilt'] <= year_range[1]) &
    (df['HouseStyle'].isin(house_styles)) &
    (df['OverallCond'] >= overall_cond_range[0]) & (df['OverallCond'] <= overall_cond_range[1])
]

st.markdown("---")

# --- Summary Metrics ---
st.subheader("Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Average Sale Price", f"${filtered_df['SalePrice'].mean():,.0f}")
col2.metric("Median Sale Price", f"${filtered_df['SalePrice'].median():,.0f}")
col3.metric("Number of Houses Displayed", filtered_df.shape[0])

st.markdown("---")

# --- Visualizations ---
st.subheader("Visualizations")

# Bar chart: Average price by neighborhood (filtered data)
avg_price_by_neigh = filtered_df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
st.bar_chart(avg_price_by_neigh)

# Histogram of SalePrice
hist_chart = alt.Chart(filtered_df).mark_bar().encode(
    alt.X('SalePrice', bin=alt.Bin(maxbins=30)),
    y='count()',
)
st.altair_chart(hist_chart, use_container_width=True)

# Scatter plot: Living Area vs Sale Price
st.subheader("Living Area vs Sale Price Scatter Plot")
scatter_chart = alt.Chart(filtered_df).mark_circle(size=60).encode(
    x='GrLivArea',
    y='SalePrice',
    color='Neighborhood',
    tooltip=['SalePrice', 'GrLivArea', 'Neighborhood']
).interactive()
st.altair_chart(scatter_chart, use_container_width=True)

st.markdown("---")

# --- Show sample data ---
st.subheader("Filtered Data Sample")
st.dataframe(filtered_df.head(10))

# --- Download filtered data ---
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(filtered_df)

st.download_button(
    label="Download filtered data as CSV",
    data=csv_data,
    file_name='filtered_house_prices.csv',
    mime='text/csv',
)

st.markdown("---")

# --- Predictive Model ---
st.header("Predict House Sale Price")

features = ['OverallQual', 'GrLivArea', 'YearBuilt', 'OverallCond', 'GarageCars']

X = df[features]
y = df['SalePrice']

model = LinearRegression()
model.fit(X, y)

# Show model performance
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
st.write(f"Model R² score: {r2:.3f}")

# Predict price from user inputs
predicted_price = model.predict([[overall_qual, gr_liv_area, year_built, overall_cond, garage_cars]])
st.subheader(f"Predicted Sale Price: ${predicted_price[0]:,.2f}")
