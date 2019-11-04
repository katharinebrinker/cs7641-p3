from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score, silhouette_samples
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.mixture import GaussianMixture

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from time import clock



class ClusterNNKMeans:

    def run_all_nn(self):

        # n_clusters = [2, 4, 6, 8, 10 ]

        k_means = KMeans(random_state = 3) # sure
        wine_data = pd.read_csv('winequality-red.csv', sep=';')
        heart_data = pd.read_csv('heart.csv', sep=',')

        print ("Data head: \n", wine_data.head())
        wine_feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
                        'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        heart_feature_cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'
        ]

        x_wine = wine_data[wine_feature_cols]  # Features
        k_means.set_params(n_clusters=2)
        k_means.fit(x_wine)
        labels = k_means.predict(x_wine)
        x_wine['cluster'] = labels

        # binary classifier; score > 5 = good
        wine_data.loc[wine_data['quality'] <= 5, 'quality'] = -1
        wine_data.loc[wine_data['quality'] > 5, 'quality'] = 1
        y_wine = wine_data.quality  # Target variable

        x_heart = heart_data[heart_feature_cols]  # Features
        k_means.set_params(n_clusters=2)
        k_means.fit(x_heart)
        labels = k_means.predict(x_heart)
        x_heart['cluster'] = labels
        y_heart = heart_data.target  # Target variable

        self.run_nn(x_wine, y_wine, "wine")
        self.run_nn(x_heart, y_heart, "heart")

        print(x_wine.head())

    @ignore_warnings(category=ConvergenceWarning)
    def run_nn(self, x_data, y_data, title):
        print("Starting nn for ", title)
        start = clock()
        nn = MLPClassifier()
        train_sizes, train_scores, test_scores = learning_curve(nn, x_data, y_data,
                                                                # Number of folds in cross-validation
                                                              cv=10,
                                                              #  Evaluation metric
                                                              scoring='accuracy',
                                                              # Use all computer cores
                                                              n_jobs=-1,
                                                              # 25 different sizes of the training set
                                                              train_sizes=np.linspace(0.01, 1.0, 25),
                                                              shuffle=True)

        end = clock()
        total_time = end - start
        print("Training ", title, " took ", total_time)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Create means and standard deviations of test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Draw lines
        plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
        plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

        # Draw bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

        # Create plot
        title_string = "Neural net learner with clustering for " + title
        plt.title(title_string)
        plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig("cluster_nn_kmeans_" + title + ".png")
        plt.clf()







ann = ClusterNNKMeans()
ann.run_all_nn()