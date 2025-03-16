# Hackathon '25  
## Time-Series Forecasting Using REST API  

### Overview  
This project is an end-to-end pipeline for **time-series forecasting** using real-world data retrieved via a REST API. The solution includes data extraction, cleaning, exploratory data analysis (EDA), model building, and deployment as an interactive web application using **Streamlit**.

---

### Problem Statement  
Time-series forecasting plays a vital role in domains such as finance, weather, sales, and healthcare. This project focuses on creating a model that can generalize to any time-series dataset and predict future values. It handles the entire workflow, from API data retrieval to the deployment of a forecasting solution.

---

### Key Features  
- **Data Retrieval**: Fetch time-series data using REST API.  
- **Data Cleaning & Preprocessing**: Handle missing values, duplicates, and time zone inconsistencies.  
- **Exploratory Data Analysis**:  
  - Plot trends, seasonality, and cyclic patterns.  
  - Identify stationarity using the Augmented Dickey-Fuller test.  
  - Analyze patterns using heatmaps and autocorrelation.  
- **Model Building**:  
  - Train ARIMA, SARIMA, Random Forest, or XGBoost models.  
  - *(Optional)* Implement LSTM or GRU for deep learning-based forecasting.  
- **Model Evaluation**:  
  - Evaluate models using MAE, RMSE, and MAPE metrics.  
  - Generate confidence intervals and provide insights.  
- **Deployment**: Deploy the solution using **Streamlit** for an interactive web interface.

---

### Workflow  
#### **Milestone 1: Data Retrieval**  
- Fetch data from a REST API.  
- Handle authentication, rate limits, and errors.  

#### **Milestone 2: Data Cleaning & Preprocessing**  
- Handle missing and duplicate data.  
- Convert timestamps to a consistent format (e.g., `YYYY-MM-DD HH:MM:SS`).  

#### **Milestone 3: Exploratory Data Analysis**  
- Visualize data trends and patterns.  
- Identify recurring cycles and stationarity.  

#### **Milestone 4: Model Building**  
- Train and fine-tune forecasting models.  

#### **Milestone 5: Forecasting**  
- Predict future values for specific timeframes (e.g., next day, week, or month).  
- Generate visualizations with confidence intervals.  

#### **Milestone 6: Deployment**  
- Build and deploy an interactive web app using **Streamlit**.  
- Provide users with an intuitive interface to upload datasets and view predictions.  

---

### Tools and Technologies  
- **Programming Language**: Python  
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scikit-learn`, `Streamlit`  

---

### How to Run the Project  
1. **Clone the repository**:  
   ```bash
   git clone <repository-url>
2. **Install dependencies:**:
   pip install -r requirements.txt
3. **Run the app:**:
   streamlit run app.py
3. **Open the web interface in your browser and interact with the deployed solution:**:
   

