import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import io

# ---- Data Loading ----
@st.cache_data
def load_data():
    """Load the cleaned dataset."""
    return pd.read_csv("cleaned_dataset4.csv")

# @st.cache_data
# def load_forecast():
#     """Load or create forecast dataset."""
#     try:
#         return pd.read_csv("5_year_forecast.csv")
#     except FileNotFoundError:
#         return pd.DataFrame()  # Empty DataFrame if no forecast data yet

# Load the data
df = load_data()

# ---- Session State for Tabs ----
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Dashboard"
if "forecast_df" not in st.session_state:
    st.session_state.forecast_df = pd.DataFrame()
if "forecast_run" not in st.session_state:
    st.session_state.forecast_run = False

# ---- Sidebar Filters ----
st.sidebar.header("Filter Options")

selected_countries = st.sidebar.multiselect("Select Countries", df["Country Name"].unique(), default=["United States"])
indicator_mode = st.sidebar.radio("Select Indicator Type", ["All Indicators", "Top Indicators"])

# Filter indicators
if indicator_mode == "Top Indicators":
    series_options = [
        "GDP (current US$)",
        "Inflation, consumer prices (annual %)",
        "Unemployment, total (% of total labor force) (modeled ILO estimate)",
        "GNI per capita, Atlas method (current US$)",
        "Exports of goods and services (% of GDP)",
        "Foreign direct investment, net inflows (% of GDP)",
        "Government expenditure on education, total (% of GDP)"
    ]
else:
    series_options = df["Series Name"].unique()

selected_series = st.sidebar.multiselect("Select Series", series_options, default=series_options[:1])
selected_years = st.sidebar.slider("Select Year Range", int(df["Year"].min()), int(df["Year"].max()), (2000, 2021))

filtered_df = df[
    (df["Country Name"].isin(selected_countries)) & 
    (df["Series Name"].isin(selected_series)) & 
    (df["Year"].between(selected_years[0], selected_years[1]))
]

# ---- Tabs Layout ----
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Forecasting", "📚 Policy Suggestions"])

# ---- Dashboard Tab ----
with tab1:
    st.session_state.active_tab = "Dashboard"
    st.title("📊 Data Dashboard")
    st.write("Explore economic and social indicators across different countries and years.")

    if not filtered_df.empty:
        for series in selected_series:
            series_df = filtered_df[filtered_df["Series Name"] == series]
            fig = px.line(series_df, x="Year", y="Value", color="Country Name", title=f"{series} Trend",
                          labels={"Value": "Indicator Value", "Year": "Year", "Country Name": "Country"})
            st.plotly_chart(fig)
    else:
        st.warning("No data available for the selected filters.")

    st.header("📂 Data Preview")
    st.dataframe(filtered_df.head(10))
    st.success("Dashboard Ready! 🚀")

    # ---- Download Button for Dashboard ----
    output_dashboard = io.BytesIO()
    with pd.ExcelWriter(output_dashboard, engine="xlsxwriter") as writer:
        filtered_df.to_excel(writer, sheet_name="Filtered Data", index=False)

    st.download_button(
        label="📥 Download Filtered Data",
        data=output_dashboard.getvalue(),
        file_name="filtered_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ---- Forecasting Functions ----
def prepare_prophet_data(df):
    """Prepare the dataset in Prophet format."""
    prophet_data = []
    
    for country in selected_countries:
        for series in selected_series:
            subset = df[(df["Country Name"] == country) & (df["Series Name"] == series)]

            if len(subset) > 5:
                subset = subset.sort_values("Year")
                subset["ds"] = pd.to_datetime(subset["Year"], format="%Y")
                subset = subset.rename(columns={"Value": "y"})[["ds", "y"]]

                prophet_data.append((country, series, subset))
    
    return prophet_data


def run_prophet_forecast(prophet_data):
    """Run Prophet forecasting using 2000-2023 data to forecast the next 10 years."""
    forecasts = []

    for country, series, data in prophet_data:
        data = data[(data["ds"] >= "2000-01-01") & (data["ds"] <= "2023-12-31")]

        if data.empty or len(data) < 5:
            continue

        model = Prophet(
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_mode='multiplicative'
        )

        model.fit(data)

        future = model.make_future_dataframe(periods=10, freq='Y')
        forecast = model.predict(future)

        forecast["Country Name"] = country
        forecast["Series Name"] = series
        forecast = forecast[["Country Name", "Series Name", "ds", "yhat", "yhat_lower", "yhat_upper"]]

        forecasts.append(forecast)

    return pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()

# ---- Forecasting Tab ----
with tab2:
    st.session_state.active_tab = "Forecasting"
    st.title("🔮 Forecasting")

    st.info("ℹ️ **For best forecast accuracy, use data from 2000 to 2023.**")

    if st.button("Run Forecast"):
        st.session_state.forecast_run = True
        prophet_data = prepare_prophet_data(filtered_df)

        if prophet_data:
            st.session_state.forecast_df = run_prophet_forecast(prophet_data)
            st.success("✅ Forecast generated successfully!")
        else:
            st.warning("Not enough data points to generate a forecast.")

    # Display forecast results
    if st.session_state.forecast_run and not st.session_state.forecast_df.empty:
        
        # ✅ Combine Historical and Forecast Data
        historical_df = filtered_df.copy()
        historical_df["Type"] = "Historical"
        
        forecast_df = st.session_state.forecast_df.copy()
        forecast_df["Type"] = "Forecasted"
        
        # ✅ Combine into one dataset for export
        combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)

        for country in selected_countries:
            for series in selected_series:
                subset = forecast_df[(forecast_df["Country Name"] == country) & 
                                     (forecast_df["Series Name"] == series)]

                if not subset.empty:
                    subset["ds"] = pd.to_datetime(subset["ds"], errors='coerce')

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(subset["ds"], subset["yhat"], 'r--', label="Forecast (yhat)")
                    ax.fill_between(subset["ds"], subset["yhat_lower"], subset["yhat_upper"], 
                                    color='pink', alpha=0.3, label="Confidence Interval")

                    ax.xaxis.set_major_locator(mdates.YearLocator(1))  
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  
                    plt.xticks(rotation=45)
                    plt.grid(visible=True, linestyle='--', alpha=0.5)
                    plt.tight_layout()

                    ax.set_title(f"{series} Forecast in {country}")
                    ax.set_xlabel("Year")
                    ax.set_ylabel("Value")
                    ax.legend()

                    st.pyplot(fig)
                    st.dataframe(subset)

        # ---- Download Button with Both Historical & Forecast Data ----
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            combined_df.to_excel(writer, sheet_name="Combined Data", index=False)

        st.download_button(
            label="📥 Download Forecast Data",
            data=output.getvalue(),
            file_name="combined_forecast_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# ---- Policy Suggestions Tab ----
with tab3:
    st.title("📚 Policy Suggestions")

    # Ensure the forecast is run first
    if not st.session_state.forecast_run:
        st.info("ℹ️ **Run the forecast first to see policy suggestions.**")
    else:
        if not st.session_state.forecast_df.empty:

            # ✅ Import Gemini only in this tab
            import google.generativeai as genai  
            
            # ✅ Configure Gemini with your API key
            # genai.configure(api_key="in-env-file")
            gemini_api_key = st.secrets["GEMINI_API_KEY"]


            policy_data = []  # To store policies for export

            def generate_gemini_policies(country, series, trend, magnitude):
                """Generate AI-powered policy suggestions using Gemini-2.0-Flash."""
                
                prompt = f"""
                You are an expert economic policy advisor.
                Based on the following data, suggest 3 detailed policies:
                
                - Country: {country}
                - Series: {series}
                - Trend: {trend} ({magnitude} change)
                
                Provide country-specific, actionable policy recommendations.
                Include short-term and long-term measures.
                """

                try:
                    # ✅ Use the `gemini-2.0-flash` model
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    response = model.generate_content(prompt)
                    
                    # ✅ Extract the generated policy suggestions
                    policies = response.text.strip()
                
                except Exception as e:
                    policies = f"⚠️ Error generating policies: {str(e)}"

                return policies

            # ✅ Generate policies for each country and series
            for country in selected_countries:
                for series in selected_series:

                    subset = st.session_state.forecast_df[
                        (st.session_state.forecast_df["Country Name"] == country) & 
                        (st.session_state.forecast_df["Series Name"] == series)
                    ]

                    if not subset.empty:
                        # ✅ Determine trend direction and magnitude
                        trend = "Rising" if subset['yhat'].iloc[-1] > subset['yhat'].iloc[0] else "Falling"

                        # Calculate magnitude of change
                        avg_change = abs(subset['yhat'].pct_change().mean())
                        magnitude = (
                            "Sharp" if avg_change > 0.05 else
                            "Moderate" if avg_change > 0.02 else
                            "Mild"
                        )

                        # ✅ Generate AI-powered policies with Gemini-2.0-Flash
                        with st.spinner(f"Generating policies for {series} in {country}..."):
                            policies = generate_gemini_policies(country, series, trend, magnitude)

                        # ✅ Display policy suggestions
                        st.subheader(f"📌 Policies for {country} - {series}")
                        st.markdown(policies)

                        # ✅ Store policies for export
                        policy_data.append({
                            "Country": country,
                            "Series": series,
                            "Trend": trend,
                            "Magnitude": magnitude,
                            "Policies": policies
                        })

            # ✅ Export to Excel
            if st.button("📥 Export Policies to Excel"):
                policy_df = pd.DataFrame(policy_data)
                policy_df.to_excel("policy_suggestions.xlsx", index=False)
                st.success("✅ Policies exported successfully!")

                with open("policy_suggestions.xlsx", "rb") as f:
                    st.download_button(
                        label="📥 Download Excel",
                        data=f,
                        file_name="policy_suggestions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.warning("No forecast data available. Run the forecast first.")
