# model_training.py
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
