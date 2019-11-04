import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
# from scipy.stats import kurtosis

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# pca = PCA(n_components="mle", svd_solver="full") # auto-guess best
ica = FastICA()
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

x_ica_wine_pre = ica.fit_transform(x_wine_scaled)
x_ica_wine_pre = pd.DataFrame(x_ica_wine_pre)

# explained_variance_wine_pre = ica.explained_variance_ratio_
print ("Wine data head (Pre-Reduction): \n", x_ica_wine_pre.head())
# print("Explained variance: \n", explained_variance_wine_pre)

x_ica_heart_pre = ica.fit_transform(x_wine_scaled)
x_ica_heart_pre = pd.DataFrame(x_ica_heart_pre)

# explained_variance_heart_pre = ica.explained_variance_ratio_
print ("Heart data head (Pre-Reduction): \n", x_ica_heart_pre.head())
# print("Explained variance: \n", explained_variance_heart_pre)

ica = FastICA(n_components=3)

x_ica_wine = ica.fit_transform(x_wine_scaled)
x_ica_wine = pd.DataFrame(x_ica_wine)

# explained_variance_wine = ica.explained_variance_ratio_
print ("Wine data head: \n", x_ica_wine.head())
# print("Explained variance: \n", explained_variance_wine)
x_ica_wine.to_csv("ica_wine.csv")

ica = FastICA(n_components=7)

x_ica_heart = ica.fit_transform(x_heart_scaled)
x_ica_heart = pd.DataFrame(x_ica_heart)

# explained_variance_heart = ica.explained_variance_ratio_
print ("Heart data head: \n", x_ica_heart.head())
# print("Explained variance: \n", explained_variance_heart)
x_ica_heart.to_csv("ica_heart.csv")

# for i in range(1, 10):
#     ica = FastICA(n_components=i)
#
#     print("------- ", i, " components -----------")
#
#     x_ica_wine = ica.fit_transform(x_wine_scaled)
#     x_ica_wine = pd.DataFrame(x_ica_wine)
#     print ("Wine data kurtosis: \n", x_ica_wine.kurt(axis = 0))
#
#     x_ica_heart = ica.fit_transform(x_heart_scaled)
#     x_ica_heart = pd.DataFrame(x_ica_heart)
#     print("Heart data kurtosis: \n", x_ica_heart.kurt(axis = 0))
#



