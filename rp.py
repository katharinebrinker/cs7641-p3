import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# pca = PCA(n_components="mle", svd_solver="full") # auto-guess best
rp = GaussianRandomProjection()
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
#
# x_rp_wine_pre = rp.fit_transform(x_wine_scaled)
# x_rp_wine_pre = pd.DataFrame(x_rp_wine_pre)
#
# # explained_variance_wine_pre = pca.explained_variance_ratio_
# print ("Wine data head (Pre-Reduction): \n", x_rp_wine_pre.head())
# # print("Explained variance: \n", explained_variance_wine_pre)
#
# x_rp_heart_pre = rp.fit_transform(x_wine_scaled)
# x_rp_heart_pre = pd.DataFrame(x_rp_heart_pre)

# explained_variance_heart_pre = rp.explained_variance_ratio_
# print ("Heart data head (Pre-Reduction): \n", x_rp_heart_pre.head())
# print("Explained variance: \n", explained_variance_heart_pre)
#
rp = GaussianRandomProjection(n_components=7)

x_rp_wine = rp.fit_transform(x_wine_scaled)
x_rp_wine = pd.DataFrame(x_rp_wine)
#
# explained_variance_wine = rp.explained_variance_ratio_
# print ("Wine data head: \n", x_rp_wine.head())
# print("Explained variance: \n", explained_variance_wine)
x_rp_wine.to_csv("rp_wine.csv")

rp = GaussianRandomProjection(n_components=9)

x_rp_heart = rp.fit_transform(x_heart_scaled)
x_rp_heart = pd.DataFrame(x_rp_heart)

# explained_variance_heart = rp.explained_variance_ratio_
# print ("Heart data head: \n", x_rp_heart.head())
# print("Explained variance: \n", explained_variance_heart)
x_rp_heart.to_csv("rp_heart.csv")

# for i in range(1, 10):
#     print("\n------- ", i, " components -----------")
#     wine_errors = []
#     heart_errors = []
#     for j in range (50):
#
#         rp = GaussianRandomProjection(n_components=i)
#
#
#
#         x_rp_wine = rp.fit_transform(x_wine_scaled)
#         x_rp_wine = pd.DataFrame(x_rp_wine)
#
#          w = rp.components_
#         p = pinv(w)
#         reconstructed = np.matmul(np.matmul(p, w), x_wine_scaled.T).T  # Unproject projected data
#         errors = np.square(x_wine_scaled - reconstructed)
#         loss_wine = np.nanmean(errors)
#         wine_errors.append(loss_wine)
#
#         if j % 10 == 0:
#             if j == 0:
#                 print("Loss (wine): \n", loss_wine)
#             else:
#                 print(loss_wine)
#
#         rp = GaussianRandomProjection(n_components=i)
#
#         x_rp_heart = rp.fit_transform(x_heart_scaled)
#         x_rp_heart = pd.DataFrame(x_rp_heart)
#
#         w = rp.components_
#         p = pinv(w)
#         reconstructed = np.matmul(np.matmul(p, w), x_heart_scaled.T).T  # Unproject projected data
#         errors = np.square(x_heart_scaled - reconstructed)
#         loss_heart = np.nanmean(errors)
#         heart_errors.append(loss_heart)
#
#         if j % 10 == 0:
#             if j == 0:
#                 print("Loss (heart): \n", loss_heart)
#             else:
#                 print(loss_heart)
#
#         # explained_variance_heart = rp.explained_variance_ratio_
#         # print ("Heart data head: \n", x_rp_heart.head())
#         # print("Explained variance (heart): \n", explained_variance_heart)
#         # x_rp_heart.to_csv("rp_heart.csv")
#     print("Average loss (wine): ", np.average(wine_errors))
#     print("Variation in loss (wine): ", np.var(wine_errors))
#     print("Average loss (heart): ", np.average(heart_errors))
#     print("Variation in loss (heart): ", np.var(heart_errors))
#
