import streamlit as st
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

st.set_page_config(page_title="House Prices Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('data/train.csv')

df = load_data()

st.title("🏠 House Prices - Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

neighborhoods = st.sidebar.multiselect(
    'Select Neighborhood(s):',
    options=df['Neighborhood'].unique(),
    default=list(df['Neighborhood'].unique())
)

year_min = int(df['YearBuilt'].min())
year_max = int(df['YearBuilt'].max())
year_range = st.sidebar.slider(
    'Year Built Range:',
    year_min, year_max, (year_min, year_max)
)

house_styles = st.sidebar.multiselect(
    "Select House Style(s):",
    options=df['HouseStyle'].unique(),
    default=list(df['HouseStyle'].unique())
)

overall_cond_range = st.sidebar.slider(
    "Overall Condition Range:",
    int(df['OverallCond'].min()),
    int(df['OverallCond'].max()),
    (int(df['OverallCond'].min()), int(df['OverallCond'].max()))
)

# Filter data
with st.spinner("Filtering data..."):
    filtered_df = df[
        (df['Neighborhood'].isin(neighborhoods)) &
        (df['YearBuilt'] >= year_range[0]) &
        (df['YearBuilt'] <= year_range[1]) &
        (df['HouseStyle'].isin(house_styles)) &
        (df['OverallCond'] >= overall_cond_range[0]) &
        (df['OverallCond'] <= overall_cond_range[1])
    ]

st.markdown(f"### Number of houses after filtering: {filtered_df.shape[0]}")

# Visualization - Average Sale Price by Neighborhood
st.subheader("Average Sale Price by Neighborhood")
avg_price_by_neigh = filtered_df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
bar_chart = alt.Chart(avg_price_by_neigh.reset_index()).mark_bar().encode(
    x=alt.X('Neighborhood', sort='-y', title='Neighborhood'),
    y=alt.Y('SalePrice', title='Average Sale Price'),
    tooltip=[alt.Tooltip('Neighborhood'), alt.Tooltip('SalePrice', format='$,.2f')]
).properties(width=700, height=400)
st.altair_chart(bar_chart, use_container_width=True)

# Histogram of Sale Prices
st.subheader("Sale Price Distribution")
hist = alt.Chart(filtered_df).mark_bar().encode(
    alt.X('SalePrice', bin=alt.Bin(maxbins=30), title='Sale Price'),
    y='count()',
    tooltip=[alt.Tooltip('count()', title='Count'), alt.Tooltip('SalePrice', title='Price')]
).properties(width=700, height=300)
st.altair_chart(hist, use_container_width=True)

# Scatter plot: Living Area vs Sale Price
st.subheader("Living Area vs Sale Price Scatter Plot")
scatter = alt.Chart(filtered_df).mark_circle(size=60).encode(
    x=alt.X('GrLivArea', title='Above Ground Living Area (sq ft)'),
    y=alt.Y('SalePrice', title='Sale Price'),
    color='Neighborhood',
    tooltip=['SalePrice', 'GrLivArea', 'Neighborhood']
).interactive().properties(width=700, height=400)
st.altair_chart(scatter, use_container_width=True)

# Prepare data for modeling
features = ['OverallQual', 'GrLivArea', 'YearBuilt', 'OverallCond', 'GarageCars']
X = df[features]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Prediction UI
st.header("Predict House Sale Price")

overall_qual = st.slider('Overall Quality (1-10)', int(df['OverallQual'].min()), int(df['OverallQual'].max()), 5)
gr_liv_area = st.slider('Living Area (sq ft)', int(df['GrLivArea'].min()), int(df['GrLivArea'].max()), 1500)
year_built = st.slider('Year Built', int(df['YearBuilt'].min()), int(df['YearBuilt'].max()), 1980)
overall_cond = st.slider('Overall Condition (1-10)', int(df['OverallCond'].min()), int(df['OverallCond'].max()), 5)
garage_cars = st.slider('Garage Cars', int(df['GarageCars'].min()), int(df['GarageCars'].max()), 1)

input_features = np.array([[overall_qual, gr_liv_area, year_built, overall_cond, garage_cars]])

with st.spinner("Predicting sale price..."):
    predicted_price = model.predict(input_features)[0]

st.markdown(f"### Predicted Sale Price: ${predicted_price:,.2f}")

# Model performance on test set
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"Model Performance on Test Set:")
st.write(f"- Root Mean Squared Error (RMSE): ${rmse:,.2f}")
st.write(f"- R-squared (R²): {r2:.3f}")

# Summary Metrics
st.subheader("Summary Metrics for Filtered Data")
avg_price = filtered_df['SalePrice'].mean()
median_price = filtered_df['SalePrice'].median()
num_houses = filtered_df.shape[0]

col1, col2, col3 = st.columns(3)
col1.metric("Average Sale Price", f"${avg_price:,.0f}")
col2.metric("Median Sale Price", f"${median_price:,.0f}")
col3.metric("Number of Houses Displayed", num_houses)
