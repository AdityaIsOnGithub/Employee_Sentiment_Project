import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sklearn.linear_model import LinearRegression
import warnings
import datetime as dt

warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('test.csv')
df['Date'] = pd.to_datetime(df['Date'])

print("Initializing AI Sentiment Pipeline...")
sentiment_analyzer = pipeline(
    "text-classification", 
    model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
    truncation=True, 
    max_length=512
)

def get_sentiment(text):
    if not isinstance(text, str) or not text.strip(): return "Neutral"
    try:
        label = sentiment_analyzer(text)[0]['label'].lower()
        if 'positive' in label: return 'Positive'
        elif 'negative' in label: return 'Negative'
        else: return 'Neutral'
    except: return "Neutral"

print("Labeling sentiment...")
df['Sentiment_Label'] = df['Message'].apply(get_sentiment)

print("Generating EDA visualizations...")
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Sentiment_Label', palette={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
plt.title("Sentiment Distribution")
plt.savefig("eda_sentiment_distribution.png")
plt.close()

print("Calculating scores, ranking, and flight risks...")
df['Month_Year'] = df['Date'].dt.to_period('M')
point_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
df['Score'] = df['Sentiment_Label'].map(point_map)

# Monthly Score Calculation & Employee Ranking
monthly_scores = df.groupby(['Employee_ID', 'Month_Year'])['Score'].sum().reset_index()
ranking = monthly_scores.groupby('Employee_ID')['Score'].mean().sort_values(ascending=False).reset_index()
ranking.to_csv("employee_ranking.csv", index=False)

# Flight Risk (>= 4 negative emails in a month)
negatives = df[df['Sentiment_Label'] == 'Negative']
monthly_neg = negatives.groupby(['Employee_ID', 'Month_Year']).size().reset_index(name='Neg_Count')
flight_risks = monthly_neg[monthly_neg['Neg_Count'] >= 4]
flight_risks.to_csv("flight_risks.csv", index=False)

print("Fitting Linear Regression Model...")
trend = df.groupby('Month_Year')['Score'].mean().reset_index()
trend['Date_Ord'] = trend['Month_Year'].dt.to_timestamp().apply(lambda x: x.toordinal())
X = trend[['Date_Ord']]
y = trend['Score']
model = LinearRegression().fit(X, y)

plt.figure(figsize=(10, 6))
plt.scatter(trend['Date_Ord'], y, color='blue', label='Actual Average Score')
plt.plot(trend['Date_Ord'], model.predict(X), color='red', label='Regression Trendline')
plt.title("Sentiment Trend Prediction")
plt.legend()
plt.savefig("sentiment_trend.png")
plt.close()

df.to_csv("test_labeled.csv", index=False)
print("Analysis complete. Check directory for generated files.")