import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import numpy as np
import streamlit as st
import pandas as pd

df = pd.read_csv("C:\\Users\\denia\\PycharmProjects\\Career Path Prediction\\job_market_data.csv")
print(df.info())
print(df.describe())

plt.figure(figsize=(10, 6))
sns.countplot(y='Job_Title', data=df, order=df['Job_Title'].value_counts().index)
plt.title("Distribution of Job Titles")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Industry', y='Salary_USD', data=df)
plt.title("Salary Distribution by Industry")
plt.xticks(rotation=90)
plt.show()

skill_counts = Counter([skill.strip() for skills in df['Required_Skills'].dropna() for skill in skills.split(',')])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(skill_counts)

plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.title("Top Skills in Demand")
plt.show()

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)

    df['Salary_USD'] = df['Salary_USD'].fillna(df['Salary_USD'].median())
    df['Job_Growth_Projection'] = df['Job_Growth_Projection'].fillna(df['Job_Growth_Projection'].mode()[0])

    df['Remote_Friendly'] = df['Remote_Friendly'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, columns=['Industry', 'Job_Growth_Projection'], drop_first=True)

    return df

df = load_and_preprocess_data("../data/job_market_data.csv")



def train_salary_growth_models(X, y_salary, y_growth):

    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    multi_regressor = MultiOutputRegressor(regressor)
    multi_classifier = MultiOutputClassifier(classifier)

    multi_regressor.fit(X, y_salary)
    multi_classifier.fit(X, y_growth)

    return multi_regressor, multi_classifier

X = df.drop(columns=['Salary_USD', 'Job_Growth_Projection_Growth'])
y_salary = df['Salary_USD']
y_growth = df['Job_Growth_Projection_Growth']

X_train, X_test, y_train_salary, y_test_salary, y_train_growth, y_test_growth = train_test_split(
    X, y_salary, y_growth, test_size=0.2, random_state=42)

salary_model, growth_model = train_salary_growth_models(X_train, y_train_salary, y_train_growth)

y_pred_salary = salary_model.predict(X_test)
print("MAE (Salary):", mean_absolute_error(y_test_salary, y_pred_salary))
print("R2 (Salary):", r2_score(y_test_salary, y_pred_salary))

y_pred_growth = growth_model.predict(X_test)
print("Accuracy (Job Growth):", accuracy_score(y_test_growth, y_pred_growth))
print("Classification Report (Job Growth):\n", classification_report(y_test_growth, y_pred_growth))


df = load_and_preprocess_data("../data/job_market_data.csv")
X, y_salary, y_growth = df.drop(columns=['Salary_USD', 'Job_Growth_Projection_Growth']), df['Salary_USD'], df['Job_Growth_Projection_Growth']

salary_model, growth_model = train_salary_growth_models(X, y_salary, y_growth)

st.title("Personalized Career Path Prediction")

job_title = st.selectbox("Job Title", df['Job_Title'].unique())
industry = st.selectbox("Industry", df['Industry'].unique())
company_size = st.number_input("Company Size", min_value=1)
location = st.selectbox("Location", df['Location'].unique())
ai_level = st.selectbox("AI Adoption Level", df['AI_Adoption_Level'].unique())
remote_friendly = st.selectbox("Remote Friendly", [1, 0])

input_data = pd.DataFrame({
    'Job_Title': [job_title], 'Industry': [industry], 'Company_Size': [company_size],
    'Location': [location], 'AI_Adoption_Level': [ai_level], 'Remote_Friendly': [remote_friendly]
})

predicted_salary = salary_model.predict(input_data)
predicted_growth = growth_model.predict(input_data)

st.write(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
st.write(f"Job Growth Projection: {'Growth' if predicted_growth[0] else 'Decline'}")