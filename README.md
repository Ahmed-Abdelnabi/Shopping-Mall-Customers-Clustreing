# Mall Customer Clustering App

## Overview
This project is a Python-based application that uses clustering algorithms to segment mall customers based on their shopping behavior. It includes scripts for model training and development of a Streamlit app, allowing users to explore customer segments.

The application provides insights into customer groups to help businesses make data-driven decisions, such as personalized marketing and resource optimization.

## Features
- Customer segmentation using clustering algorithms (e.g., K-Means, Hierarchical Clustering).
- User-friendly Streamlit application that allows customers to input their data and see which group they belong to.
- Visualizations of clusters and customer behavior.
- Models built with data from the **Mall Customer Segmentation** dataset.

## Scripts
1. **Mall Customer Clustering Models.py**
   - This script trains multiple clustering models, including K-Means and Hierarchical Clustering, on mall customer data.
   - It generates clusters based on features such as Age, Annual Income, and Spending Score.
   - Key visualizations (e.g., elbow plots, dendrograms) are included for model evaluation and cluster analysis.

2. **Clustering_app.py**
   - This is the Streamlit app script that allows users to interact with the trained clustering model.
   - The app takes customer input (such as age, income, and spending habits) and outputs the cluster they belong to.
   - Built using Streamlit for a simple and interactive user experience.

## Installation
To run the app locally, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mall-customer-clustering-app.git
cd mall-customer-clustering-app
