# 🏠 House Price Prediction Dashboard

An interactive data science dashboard built with **Streamlit**, powered by machine learning models to explore and predict house sale prices. Designed with Agile principles, this project demonstrates iterative development, clean UI/UX, and collaboration features.

---

## 🔍 Overview

This application visualizes housing market trends using the popular **House Prices - Advanced Regression Techniques** dataset from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). Users can:

- Filter data by neighborhood, year built, house style, and condition
- View insightful visualizations (bar chart, histogram, scatter plot)
- Predict house prices using Linear Regression
- Compare model performance (Linear Regression vs Random Forest)
- Download filtered datasets

---

## 📦 Features

✅ Sidebar filters for dynamic exploration  
✅ Predictive model using user inputs  
✅ Visualizations powered by Altair  
✅ Model comparison with performance metrics (RMSE, R²)  
✅ Streamlit Cloud deployment ready  
✅ Lightweight and responsive UI

---

# 🔧 Tech Stack

- Python 🐍
- Streamlit (dashboard)
- Scikit-learn (machine learning)
- Pandas (data manipulation)
- Altair (visualization)
- Git/GitHub (version control & collaboration)

---

# 📂 Project Structure

data-science-dashboard/

│

├── data/ # Dataset (e.g., train.csv)

├── src/ # (Optional) Custom modules if added

├── tests/ # (Optional) Unit tests

├── dashboard.py # Main Streamlit app

├── requirements.txt # Python dependencies

└── README.md # Project overview and instructions

---

# ▶️ Run the App Locally

**# Clone the repository**

git clone https://github.com/tenglexin/data-science-dashboard.git
cd data-science-dashboard

**# Install dependencies**

pip install -r requirements.txt

**# Run the dashboard**

streamlit run dashboard.py

---

# 🌐 Live Demo

🌐 Live App: [Streamlit Cloud App](https://house-price-predictor-dashboard.streamlit.app/)  
📂 GitHub Repo: [GitHub Repository](https://github.com/tenglexin/data-science-dashboard)

---

# 📈 Model Info

The app uses a Linear Regression model trained on key features like:

- Overall Quality
- Living Area
- Year Built
- Overall Condition
- Garage Capacity

Performance on test set:

- R²: Measures how well the model fits the data
- RMSE: Root Mean Squared Error shows prediction error

---

# 🛠️ Agile Development

This app was developed using Agile principles:

- MVD built first, then improved over iterations
- Sprint log tracks user stories, backlog, and improvements
- Frequent commits showing iterative development

---

# 🤝 Collaborators

- tenglexin
- leeseeping

---

# 📜 License

This project is for educational use. Data provided by Kaggle.
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv
