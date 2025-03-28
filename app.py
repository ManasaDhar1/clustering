#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib
import os

app = Flask(__name__)

df = pd.read_csv("World_Development_Dataset.csv")

# Load the trained KMeans model
kmeans = joblib.load("model.pkl")

# Get column names for feature selection
columns = df.columns.tolist()

@app.route('/')
def index():
    return render_template('index.html', columns=columns)

@app.route('/cluster', methods=['POST'])
def cluster():
    selected_features = request.form.getlist('features')
    
    if not selected_features:
        return "Please select at least one feature."
    
    data = df[selected_features].dropna()
    data['Cluster'] = kmeans.predict(data)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(data[selected_features[0]], data[selected_features[1]], c=data['Cluster'], cmap='viridis')
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.title('K-Means Clustering')
    plt.colorbar()
    plt.savefig("static/cluster_plot.png")
    
    return render_template('result.html', image='static/cluster_plot.png')

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




