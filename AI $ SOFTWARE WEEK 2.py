#!/usr/bin/env python
# coding: utf-8

# ## AI for Sustainable Development – SDG13: Climate Action

# In[8]:


# Forecasting CO2 Emissions using Machine Learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset

df = pd.read_csv(r"D:\CO2 emission by countries.csv", encoding='latin1')

# Rename columns for easier handling
df.rename(columns={
    'CO2 emission (Tons)': 'CO2_Emissions',
    'Population(2022)': 'Population',
    'Country Code': 'Country'
}, inplace=True)

# Display a sample
print(df.head())


# In[9]:


# 2. Data preprocessing

# Ensure numeric types
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['CO2_Emissions'] = pd.to_numeric(df['CO2_Emissions'], errors='coerce')
df['Population'] = pd.to_numeric(df['Population'], errors='coerce')

# Remove missing or invalid data
df.dropna(subset=['Year', 'CO2_Emissions', 'Population'], inplace=True)

# Create previous year's emissions per country
df['Prev_Year_Emissions'] = df.groupby('Country')['CO2_Emissions'].shift(1)

# Drop rows without previous emissions
df.dropna(subset=['Prev_Year_Emissions'], inplace=True)

# Select features and target
X = df[['Year', 'Population', 'Prev_Year_Emissions']]
y = df['CO2_Emissions']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[10]:


# 3. Train models

lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)


# In[11]:


# 4. Evaluate models

print("\nModel Performance:\n")
for model, name in zip([lr, rf], ['Linear Regression', 'Random Forest']):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} -> MAE: {mae:.3f}, R²: {r2:.3f}")


# ## Visualizations & Insights

# In[14]:


# 1. Top Emitting Countries

top_emitters = df.groupby('Country')['CO2_Emissions'].mean().nlargest(10)
plt.figure(figsize=(10,6))
top_emitters.plot(kind='bar', color='salmon')
plt.title("Top 10 Countries by Average CO₂ Emissions")
plt.ylabel("Average CO₂ Emissions (Tons)")
plt.xlabel("Country")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[15]:


# 2. Emissions Trend Over Time (Global)

global_trend = df.groupby('Year')['CO2_Emissions'].mean()
plt.figure(figsize=(10,5))
plt.plot(global_trend.index, global_trend.values, marker='o')
plt.title("Global Average CO₂ Emissions Over Time")
plt.xlabel("Year")
plt.ylabel("Average CO₂ Emissions (Tons)")
plt.grid(True)
plt.show()


# In[18]:


# 4. Population vs. Emissions

plt.figure(figsize=(8,6))
plt.scatter(df['Population'], df['CO2_Emissions'], alpha=0.6)
plt.title("Population vs CO₂ Emissions")
plt.xlabel("Population (2022)")
plt.ylabel("CO₂ Emissions (Tons)")
plt.xscale('log')  # optional, for better visibility if population varies widely
plt.grid(True)
plt.show()



# In[19]:


y_pred_all = rf.predict(X)
plt.figure(figsize=(6,6))
plt.scatter(y, y_pred_all, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Actual vs Predicted CO₂ Emissions")
plt.xlabel("Actual Emissions")
plt.ylabel("Predicted Emissions")
plt.grid(True)
plt.show()


# In[12]:


# Visualization for one country

sample_country = 'Kenya'  # change this to any other country name
country_data = df[df['Country'] == sample_country].copy()

if not country_data.empty:
    # Generate predictions for the selected country
    country_data['Predicted'] = rf.predict(country_data[['Year', 'Population', 'Prev_Year_Emissions']])

    plt.figure(figsize=(8,5))
    plt.plot(country_data['Year'], country_data['CO2_Emissions'], label='Actual', marker='o')
    plt.plot(country_data['Year'], country_data['Predicted'], label='Predicted', linestyle='--', marker='x')
    plt.title(f"CO₂ Emissions Forecast - {sample_country}")
    plt.xlabel('Year')
    plt.ylabel('CO₂ Emissions (Tons)')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print(f"No data found for {sample_country}")


# In[ ]:




