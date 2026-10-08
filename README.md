# Employee Sentiment & Engagement Analysis

## Methodology
*   **Sentiment Labeling:** Uses the `twitter-roberta-base-sentiment-latest` LLM (PyTorch backend) to classify text into Positive, Negative, or Neutral.
*   **Scoring & Ranking:** Maps sentiments (+1, -1, 0) and aggregates them monthly. Ranks employees by highest average monthly score.
*   **Flight Risk:** Flags employees sending 4 or more negative messages in a single month.
*   **Predictive Modeling:** Uses `sklearn` Linear Regression on ordinal time-series data to plot overall company sentiment trends.

## Setup and Usage
1. Install dependencies: `pip install pandas numpy matplotlib seaborn transformers torch scikit-learn`
2. Run `python generate_data.py` to create the initial dataset.
3. Run `python sentiment_analysis.py` to output the final analytics, CSVs, and visualization PNGs.