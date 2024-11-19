# Automobile Customer Segmentation (Clustering) Project
## Context
Customer segmentation is the practice of dividing a customer base into groups of individuals that are similar 
in specific ways relevant to marketing, such as age, gender, interests and spending habits.
Companies employing customer segmentation operate under the fact that every customer is different and that their 
marketing efforts would be better served if they target specific, smaller groups with messages that those consumers 
would find relevant and lead them to buy something. Companies also hope to gain a deeper understanding of their 
customers' preferences and needs with the idea of discovering what each segment finds most valuable to more 
accurately tailor marketing materials toward that segment.
Content
## Content
An automobile company has plans to enter new markets with their existing products (P1, P2, P3, P4 and P5). 
After intensive market research, they’ve deduced that the behavior of new market is similar to their existing market.
In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they 
performed segmented outreach and communication for different segment of customers. This strategy has work 
exceptionally well for them. They plan to use the same strategy on new markets and have identified 2627 new potential 
customers.
The dataset provides the details of the existing and potential customers of the company based on the purchase 
history and the corresponding segments they have been classified into.
## Variable description
- `CustomerID` : unique customer ID
- `Gender` : gender of the customer
- `Married` : marital status of the customer
- `Age` : age of the customer
- `Graduated` : specifies whether the customer a graduate?
- `Profession` : profession of the customer
- `WorkExperience` : work experience of the customer in years
- `SpendingScore` : spending score of the customer
- `FamilySiz`e : number of family members of the customer (including the customer)
- `Category` : anonymised category for the customer
- `Segmentation` : (target variable) customer segment of the customer
## Methods Used
Clustering is an unsupervised machine-learning technique. It is the process of division of the dataset into 
groups in which the members in the same group possess similarities in features. \
k-means clustering is a method of vector quantization, that aims to partition n observations into 
k clusters in which each observation belongs to the cluster with the nearest mean (cluster centers 
or cluster centroid), serving as a prototype of the cluster. This results in a partitioning 
of the data space into Voronoi cells. k-means clustering minimizes within-cluster variances 
(squared Euclidean distances), but not regular Euclidean distances.
The commonly used clustering techniques are:
- K-Means clustering, 
- Hierarchical clustering, 
- Density-based clustering, 
- Model-based clustering. 
We can implement the K-Means clustering machine learning algorithm in the elbow method using 
the scikit-learn library in Python.
## Main aspects
- Elbow method to determine the optimal number of clusters in K-Means using Python;
- Assess the quality of K-Means clusters using Within-Cluster Sum of Squares (WCSS).
## Dimensionality Reduction with Principal Component Analysis (PCA)
PCA is a technique used to emphasize variation and capture strong patterns in a dataset. It transforms the data 
into a new set of variables, the principal components, which are orthogonal (uncorrelated), ensuring that 
the first principal component captures the most variance, and each succeeding one, less so. This transformation 
is not just a mere reduction of dimensions; it’s an insightful distillation of data.
**In the following sections, we will cover:**
- Pre-processing: Preparing your dataset for analysis.
- Scaling: Why and how to scale your data.
- Optimal PCA Components: Determining the right number of components.
- Applying PCA: Transforming your data.
- KMeans Clustering: Grouping the transformed data.
- Analyzing PCA Loadings: Understanding what your components represent.
- From PCA Space to Original Space: Interpreting the cluster centers.
- Centroids and Means: Comparing cluster centers with the original data mean.
- Deep Dive into Loadings: A closer look at the features influencing each principal component.
## Hierarchical clustering
Hierarchical clustering, also known as hierarchical cluster analysis, is an algorithm that groups similar objects into 
groups called clusters.
## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Creates cluster for noise. Density is key.
## Silhouette Score
The overall silhouette score is the average silhouette score for all points in the dataset. It provides a single 
measure of the overall clustering quality.
## Davies-Bouldin score
The score is defined as the average similarity measure of each cluster with its most similar cluster, where similarity 
is the ratio of within-cluster distances to between-cluster distances. Thus, clusters which are farther apart and less 
dispersed will result in a better score.
## Dunn index
The Dunn index (DI) (introduced by J. C. Dunn in 1974), a metric for evaluating clustering algorithms, is an internal 
evaluation scheme, where the result is based on the clustered data itself. Like all other such indices, the aim of this 
Dunn index to identify sets of clusters that are compact, with a small variance between members of the cluster, and 
well separated, where the means of different clusters are sufficiently far apart, as compared to the within cluster 
variance. 
Higher the Dunn index value, better is the clustering. The number of clusters that maximizes Dunn index is taken as the 
optimal number of clusters k. It also has some drawbacks. As the number of clusters and dimensionality of the data 
increase, the computational cost also increases. 
## Requirements
1. Pasirinkti duomenų rinkinį.
2. Sutvarkyti gautus duomenis.
3. Įgyvendinti tris klasterizavimo metodus: K-means, DBSCAN ir Agglomerative.
4. K-means ir Agglomerative metoduose naudoti alkūnės metodą optimaliam klasterių skaičiui nustatyti.
5. Agglomerative metodui - nubraižyti dendrogramą.
6. Nustayti geriausius parametrus.
7. Gautiems sprendimams įvertinti taikyti tris klasterizavimo kokybės metrikas: Silhouette index, Davies-Bouldin score, Dunn score.
8. Nubraižyti gautus klasterius.
9. Išsiaiškinti, kodėl jūsų atveju vienas ar kitas algoritmas veikia geriau.
10. Pristatyti rezultatus.
## Resources
![Automobile Customer Segmentation Dataset](https://www.kaggle.com/datasets/akashdeepkuila/automobile-customer)

