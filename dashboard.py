import streamlit as st
import pandas as pd
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('data/train.csv')

# --- Filters ---
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

filtered_df = df[
    (df['Neighborhood'].isin(neighborhoods)) &
    (df['YearBuilt'] >= year_range[0]) &
    (df['YearBuilt'] <= year_range[1])
]

st.write(f"Number of houses after filtering: {filtered_df.shape[0]}")

# --- Visualization ---
avg_price_by_neigh = filtered_df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
st.bar_chart(avg_price_by_neigh)

chart = alt.Chart(filtered_df).mark_bar().encode(
    alt.X('SalePrice', bin=alt.Bin(maxbins=30)),
    y='count()',
)
st.altair_chart(chart, use_container_width=True)

# --- Predictive Model ---

features = ['OverallQual', 'GrLivArea', 'YearBuilt']
X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

st.header("Predict House Sale Price")

overall_qual = st.slider('Overall Quality (1-10)', int(df['OverallQual'].min()), int(df['OverallQual'].max()), 5)
gr_liv_area = st.slider('Living Area (sq ft)', int(df['GrLivArea'].min()), int(df['GrLivArea'].max()), 1500)
year_built = st.slider('Year Built', int(df['YearBuilt'].min()), int(df['YearBuilt'].max()), 1980)

predicted_price = model.predict([[overall_qual, gr_liv_area, year_built]])
st.write(f"Predicted Sale Price: ${predicted_price[0]:,.2f}")
