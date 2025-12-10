#  Prodigy ML – Task 2  
## Customer Segmentation & Recommendation System (Clustering + Market Basket Analysis)

This project is a part of my Machine Learning Internship at **Prodigy InfoTech**.  
In this task, I performed **Customer Segmentation using K-Means Clustering** and built a **Recommendation System with Apriori Association Rules**.

###  Project Overview

 **Customer Segmentation:**  
Using clustering (K-Means), similar customers are grouped based on purchase behavior, helping businesses target marketing strategies efficiently.

 **Product Recommendation System:**  
Using the **Apriori Algorithm** & **Association Rules**, frequently bought items are discovered and used for recommendation insights.

---

###  Tech Stack & Libraries Used

| Area | Tools |
|------|-------|
| Programming | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML Algorithms | Scikit-Learn |
| Recommendation | mlxtend (Apriori, Association Rules) |

---

### 📂 Key Features

✔ Perform **data preprocessing & feature scaling**  
✔ Apply **K-Means Clustering** and choose optimal clusters  
✔ Plot **Cluster Groups using Visualization**  
✔ Generate **frequent itemsets using Apriori**  
✔ Extract **recommendation rules (Support, Confidence, Lift)**  
✔ Provide actionable business insights 🛒📊

---

### 📊 Sample Workflow

```python
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

# Example: K-Means Clustering Model
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(data)

# Example: Market Basket Algorithm
frequent_items = apriori(df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
