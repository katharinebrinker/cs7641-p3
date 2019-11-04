import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# lda = lda(n_components="mle", svd_solver="full") # auto-guess best
lda = LinearDiscriminantAnalysis()
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

x_wine_scaled = (wine_data-wine_data.mean())/wine_data.std()
x_wine_scaled = pd.DataFrame(x_wine_scaled)

x_heart_scaled = (heart_data-heart_data.mean())/heart_data.std()
x_heart_scaled = pd.DataFrame(x_heart_scaled)

# x_wine_scaled = x_wine
# x_heart_scaled = x_heart

# print(x_wine_scaled.head())

x_lda_wine_pre = lda.fit_transform(x_wine_scaled, y_wine)
x_lda_wine_pre = pd.DataFrame(x_lda_wine_pre)

explained_variance_wine_pre = lda.explained_variance_ratio_
print ("Wine data head (Pre-Reduction): \n", x_lda_wine_pre.head())
print("Explained variance: \n", explained_variance_wine_pre)

x_lda_heart_pre = lda.fit_transform(x_heart_scaled, y_heart)
x_lda_heart_pre = pd.DataFrame(x_lda_heart_pre)

explained_variance_heart_pre = lda.explained_variance_ratio_
print ("Heart data head (Pre-Reduction): \n", x_lda_heart_pre.head())
print("Explained variance: \n", explained_variance_heart_pre)
#
lda = LinearDiscriminantAnalysis(n_components=4, solver="eigen")

x_lda_wine = lda.fit_transform(x_wine_scaled, y_wine)
x_lda_wine = pd.DataFrame(x_lda_wine)

explained_variance_wine = lda.explained_variance_ratio_
print ("Wine data head: \n", x_lda_wine.head())
print("Explained variance: \n", explained_variance_wine)
x_lda_wine.to_csv("lda_wine.csv")

lda = LinearDiscriminantAnalysis(n_components=1, solver="eigen")

x_lda_heart = lda.fit_transform(x_heart_scaled, y_heart)
x_lda_heart = pd.DataFrame(x_lda_heart)

explained_variance_heart = lda.explained_variance_ratio_
print ("Heart data head: \n", x_lda_heart.head())
print("Explained variance: \n", explained_variance_heart)
x_lda_heart.to_csv("lda_heart.csv")

# for i in range(1,10):
#     lda = LinearDiscriminantAnalysis(n_components=i, solver="eigen")
#
#     print("------- ", i, " components -----------")
#
#     x_lda_wine = lda.fit_transform(x_wine_scaled, y_wine)
#     x_lda_wine = pd.DataFrame(x_lda_wine)
#
#     explained_variance_wine = lda.explained_variance_ratio_
#     # print ("Wine data head: \n", x_lda_wine.head())
#     print("Explained variance (wine): \n", explained_variance_wine)
#     # x_lda_wine.to_csv("lda_wine.csv")
#
#     lda2 = LinearDiscriminantAnalysis(n_components=i, solver="eigen")
#     x_lda_heart = lda2.fit_transform(x_heart_scaled, y_heart)
#     x_lda_heart = pd.DataFrame(x_lda_heart)
#
#     explained_variance_heart = lda2.explained_variance_ratio_
#     # print ("Heart data head: \n", x_lda_heart.head())
#     print("Explained variance (heart): \n", explained_variance_heart)
#     # x_lda_heart.to_csv("lda_heart.csv")
#
