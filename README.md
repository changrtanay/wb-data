# Data Cleaning, Forecasting, and Policy Analysis
# On World Bank's World Development Indicators Dataset

## Overview
This project involves data cleaning, transformation, forecasting, and policy analysis using economic and social indicators from multiple countries. The workflow includes:
- **Data Preprocessing:** Cleaning and reshaping the dataset.
- **Dashboard:** Interactive visualization using Streamlit.
- **Forecasting:** Future trend predictions using Facebook Prophet.
- **Policy Suggestions:** AI-generated policy recommendations based on forecasted trends.

## Features
### 1. **Data Preprocessing**
- Loads raw dataset (`dataset.csv`) and handles missing values.
- Removes metadata rows and columns with excessive missing values.
- Reshapes data into a long format for easier analysis.
- Converts years to integers and interpolates missing values.
- Performs forward and backward filling to handle remaining missing values.
- Applies log transformation for normalization.
- Saves cleaned data as `cleaned_dataset4.csv`.

### 2. **Interactive Dashboard**
- Built with **Streamlit** to visualize trends in economic and social indicators.
- Provides country and indicator selection filters.
- Displays interactive line charts using **Plotly**.
- Allows users to download filtered datasets in Excel format.

### 3. **Forecasting with Prophet**
- Implements **Facebook Prophet** for time series forecasting.
- Uses historical data (2000-2023) to predict the next 10 years.
- Shows forecasted trends with confidence intervals.
- Supports multiple countries and indicators.
- Users can download combined historical and forecasted data.

### 4. **AI-Powered Policy Suggestions**
- Uses **Google Gemini AI** to generate policy recommendations.
- Provides country-specific, actionable policy insights.
- Generates short-term and long-term economic policies.
- Users can export policy recommendations as an Excel file.

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Required libraries: `pandas`, `numpy`, `streamlit`, `plotly`, `prophet`, `matplotlib`

### Installation
```sh
pip install pandas numpy streamlit plotly prophet matplotlib openpyxl
```

### Running the Application
```sh
streamlit run app.py
```

## File Structure
```
├── 5_year_forecast.csv         # Forecasted Data
├── cleaned_dataset4.csv        # Cleaned dataset after preprocessing
├── dashboard2.py               # Streamlit application
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Required dependencies
```

## Usage
1. **Run the Streamlit app**: `streamlit run app.py`.
2. **Filter data**: Select countries, indicators, and year range.
3. **View Trends**: Explore interactive charts in the Dashboard tab.
4. **Forecast Future Trends**: Click "Run Forecast" in the Forecasting tab.
5. **Generate Policy Insights**: View AI-generated policy recommendations.
6. **Download Reports**: Export filtered data, forecasts, and policies as Excel files.

## Credits
- **Data Cleaning & Transformation:** `pandas`, `numpy`
- **Dashboard & Visualization:** `streamlit`, `plotly`
- **Forecasting Model:** `Facebook Prophet`
- **Policy Analysis:** `Google Gemini AI`

## License
This project is licensed under the MIT License.

