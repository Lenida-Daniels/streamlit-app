import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


# Load your data
#@st.cache_data
def load_data():
    df = pd.read_csv("refined_subset_df.csv") 
    df.columns = df.columns.str.lower().str.strip()
    return df

df = load_data()


# Title and description
st.title("🌾 Climate Impact on Crop Yields in East Africa (2000–2023)")



st.markdown("""
This dashboard explores how **temperature**, **precipitation**, and **humidity** affect **crop yield** across East African countries.
""")


# Sidebar filters
countries = df['country'].unique()
crops = df['crop'].unique()

selected_country = st.sidebar.selectbox("Select Country", countries)
selected_crop = st.sidebar.selectbox("Select Crop", crops)

subset = df[(df['country'] == selected_country) & (df['crop'] == selected_crop)]

# Line chart: Climate over years
st.subheader(f"Climate Trends in {selected_country} for {selected_crop}")
fig, ax = plt.subplots()
sns.lineplot(data=subset, x='year', y='temperature', label='Temperature (°C)', ax=ax)
sns.lineplot(data=subset, x='year', y='precipitation', label='Precipitation (mm)', ax=ax)
sns.lineplot(data=subset, x='year', y='humidity', label='Humidity (%)', ax=ax)
plt.legend()
st.pyplot(fig)

# Yield vs climate factors
st.subheader(f"📊 Yield vs Climate in {selected_country} for {selected_crop}")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Yield vs Temperature**")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=subset, x='temperature', y='yield', ax=ax1)
    st.pyplot(fig1)

with col2:
    st.markdown("**Yield vs Precipitation**")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=subset, x='precipitation', y='yield', ax=ax2)
    st.pyplot(fig2)

with col3:
    st.markdown("**Yield vs Humidity**")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=subset, x='humidity', y='yield', ax=ax3)
    st.pyplot(fig3)

# Correlation across countries
st.subheader("📉 Correlation of Climate Variables with Yield (by Country)")

#@st.cache_data
def climate_yield_corr_by_country(df):
    results = []
    for country in df['country'].unique():
        temp = df[df['country'] == country]
        if len(temp) >= 5:
            results.append({
                'country': country,
                'temp_vs_yield': temp['temperature'].corr(temp['yield']),
                'precip_vs_yield': temp['precipitation'].corr(temp['yield']),
                'humidity_vs_yield': temp['humidity'].corr(temp['yield']),
            })
    return pd.DataFrame(results)

corr_df = climate_yield_corr_by_country(df)

metric = st.selectbox("Select Metric", ['temp_vs_yield', 'precip_vs_yield', 'humidity_vs_yield'])

fig_corr, ax_corr = plt.subplots(figsize=(8, 5))
sns.barplot(data=corr_df.sort_values(metric), x=metric, y='country', ax=ax_corr)
ax_corr.set_title(f"Correlation: {metric.replace('_', ' ').title()}")
st.pyplot(fig_corr)


# Prepare data
features = df[["temperature", "precipitation", "humidity"]]
target = df["yield"]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression().fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
rf_preds = rf.predict(X_test)



df["Predicted_Yield"] = rf.predict(features)
df["Yield_Error"] = abs(df["Predicted_Yield"] - df["yield"])
df["Climate_Risk_Score"] = df["Yield_Error"] * df[["temperature", "precipitation"]].std(axis=1)
df["Climate_Risk_Score"] = (df["Climate_Risk_Score"] - df["Climate_Risk_Score"].min()) / \
                           (df["Climate_Risk_Score"].max() - df["Climate_Risk_Score"].min())



country_risk = df.groupby("country")["Climate_Risk_Score"].mean().reset_index()

fig = px.choropleth(country_risk, locations="country", locationmode="country names",
                    color="Climate_Risk_Score", color_continuous_scale="Reds",
                    title="Climate Risk Score by Country",
                    width=1000, height=600)

st.plotly_chart(fig, use_container_width=True)
# Optional: Show filtered data
if st.checkbox("Show Filtered Data"):
    st.dataframe(subset)
