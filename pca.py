import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# pca = PCA(n_components="mle", svd_solver="full") # auto-guess best
pca = PCA()
wine_data = pd.read_csv('winequality-red.csv', sep=';')
heart_data = pd.read_csv('heart.csv', sep=',')

# print ("Data head: \n", wine_data.head())
wine_feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
heart_feature_cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'
]

x_wine = wine_data[wine_feature_cols]  # Features
# binary classifier; score > 5 = good
# wine_data.lococ[wine_data['quality'] <= 5, 'quality'] = -1
# wine_data.loc[wine_data['quality'] > 5, 'quality'] = 1
y_wine = wine_data.quality  # Target variable

x_heart = heart_data[heart_feature_cols]  # Features
y_heart = heart_data.target  # Target variable

x_wine_scaled = StandardScaler().fit_transform(x_wine)
x_wine_scaled = pd.DataFrame(x_wine_scaled)

x_heart_scaled = StandardScaler().fit_transform(x_heart)
x_heart_scaled = pd.DataFrame(x_heart_scaled)

x_pca_wine_pre = pca.fit_transform(x_wine_scaled)
x_pca_wine_pre = pd.DataFrame(x_pca_wine_pre)

explained_variance_wine_pre = pca.explained_variance_ratio_
print ("Wine data head (Pre-Reduction): \n", x_pca_wine_pre.head())
print("Explained variance: \n", explained_variance_wine_pre)

x_pca_heart_pre = pca.fit_transform(x_heart_scaled)
x_pca_heart_pre = pd.DataFrame(x_pca_heart_pre)

explained_variance_heart_pre = pca.explained_variance_ratio_
print ("Heart data head (Pre-Reduction): \n", x_pca_heart_pre.head())
print("Explained variance: \n", explained_variance_heart_pre)
#
# pca = PCA(n_components=4)
#
# x_pca_wine = pca.fit_transform(x_wine_scaled)
# x_pca_wine = pd.DataFrame(x_pca_wine)
#
# explained_variance_wine = pca.explained_variance_ratio_
# print ("Wine data head: \n", x_pca_wine.head())
# print("Explained variance: \n", explained_variance_wine)
# x_pca_wine.to_csv("pca_wine.csv")
#
# pca = PCA(n_components=2)
#
# x_pca_heart = pca.fit_transform(x_heart_scaled)
# x_pca_heart = pd.DataFrame(x_pca_heart)
#
# explained_variance_heart = pca.explained_variance_ratio_
# print ("Heart data head: \n", x_pca_heart.head())
# print("Explained variance: \n", explained_variance_heart)
# x_pca_heart.to_csv("pca_heart.csv")

for i in range(10):
    pca = PCA(n_components=i)

    print("------- ", i, " components -----------")

    x_pca_wine = pca.fit_transform(x_wine_scaled)
    x_pca_wine = pd.DataFrame(x_pca_wine)

    explained_variance_wine = pca.explained_variance_ratio_
    # print ("Wine data head: \n", x_pca_wine.head())
    print("Explained variance (wine): \n", explained_variance_wine)
    # x_pca_wine.to_csv("pca_wine.csv")

    x_pca_heart = pca.fit_transform(x_heart_scaled)
    x_pca_heart = pd.DataFrame(x_pca_heart)

    explained_variance_heart = pca.explained_variance_ratio_
    # print ("Heart data head: \n", x_pca_heart.head())
    print("Explained variance (heart): \n", explained_variance_heart)
    # x_pca_heart.to_csv("pca_heart.csv")

