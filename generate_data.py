import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generates synthetic employee messages
np.random.seed(42)
dates = [datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(500)]
emp_ids = np.random.randint(100, 110, 500) # Small pool to ensure multiple messages per month
messages = [
    "I love the new project, it's very exciting!", "The management is terrible and I feel burnt out.",
    "The meeting was okay, nothing special.", "I am learning a lot but the hours are incredibly long.",
    "Fantastic work environment and great benefits.", "I want to quit, this is far too stressful.",
    "Neutral experience so far.", "Highly recommend this company to others.",
    "Poor work-life balance.", "Good overall experience this month."
]

data = [[dates[i], emp_ids[i], np.random.choice(messages)] for i in range(500)]
df = pd.DataFrame(data, columns=['Date', 'Employee_ID', 'Message'])
df.to_csv('test.csv', index=False)
print("Unlabeled test.csv generated successfully.")