# data-science-dashboard
Agile-based Streamlit dashboard
🏠 House Price Prediction Dashboard
An interactive Streamlit dashboard for exploring and predicting house prices using the House Prices - Advanced Regression Techniques dataset.

Built as part of a data science project using Agile principles, iterative sprints, and a Minimum Viable Dashboard (MVD) approach.

🚀 Features
Sidebar Filters: Filter data by neighborhood, year built, house style, and overall condition

Visualizations:

Average price by neighborhood

Sale price distribution

Scatter plot: living area vs. price

Predictive Model:

Input home features to predict sale price using Linear Regression

Shows model performance: RMSE and R²

Summary Metrics: Displays average price, median price, and total listings after filters

🔧 Tech Stack
Python 🐍

Streamlit (dashboard)

Scikit-learn (machine learning)

Pandas (data manipulation)

Altair (visualization)

Git/GitHub (version control & collaboration)

📂 Project Structure
bash
Copy
Edit
data-science-dashboard/
│
├── data/               # Dataset (e.g., train.csv)
├── src/                # (Optional) Custom modules if added
├── tests/              # (Optional) Unit tests
├── dashboard.py        # Main Streamlit app
├── requirements.txt    # Python dependencies
└── README.md           # Project overview and instructions
▶️ Run the App Locally
Make sure you have Python and pip installed.

bash
Copy
Edit
# Clone the repository
git clone https://github.com/tenglexin/data-science-dashboard.git
cd data-science-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard.py
🌐 Live Demo
Click here to view the live app on Streamlit Cloud
https://house-price-predictor-dashboard.streamlit.app/

📈 Model Info
The app uses a Linear Regression model trained on key features like:

Overall Quality

Living Area

Year Built

Overall Condition

Garage Capacity

Performance on test set:

R²: Measures how well the model fits the data

RMSE: Root Mean Squared Error shows prediction error

🛠️ Agile Development
This app was developed using Agile principles:

MVD built first, then improved over iterations

Sprint log tracks user stories, backlog, and improvements

Frequent commits showing iterative development

🤝 Collaborators
tenglexin
leeseeping


📜 License
This project is for educational use. Data provided by Kaggle.
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=train.csv 

