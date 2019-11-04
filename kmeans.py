import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score, silhouette_samples
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from collections import defaultdict

class KMeansClustering:

    k_means = None
    n_clusters = []

    def __init__(self):
        self.k_means = KMeans(random_state = 3) # sure
        # 2 for good / bad, (padding), 10 features / scores
        self.n_clusters = [2, 4, 6, 8, 10 ]

    def run_all_kmeans(self):

        # n_clusters = [2, 4, 6, 8, 10 ]

        # k_means = KMeans(random_state = 3) # sure
        wine_data = pd.read_csv('winequality-red.csv', sep=';')
        heart_data = pd.read_csv('heart.csv', sep=',')
        wine_data_pca = pd.read_csv("pca_wine.csv")
        heart_data_pca = pd.read_csv("pca_heart.csv")
        wine_data_ica = pd.read_csv("ica_wine.csv")
        heart_data_ica = pd.read_csv("ica_heart.csv")
        wine_data_rp = pd.read_csv("rp_wine.csv")
        heart_data_rp = pd.read_csv("rp_heart.csv")
        wine_data_lda = pd.read_csv("lda_wine.csv")
        heart_data_lda = pd.read_csv("lda_heart.csv")

        print ("Data head: \n", wine_data.head())
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

        self.run_kmeans(x_wine, y_wine, "wine")
        self.run_kmeans(x_heart, y_heart, "heart")
        self.run_kmeans(wine_data_pca, y_wine, "wine_pca")
        self.run_kmeans(heart_data_pca, y_heart, "heart_pca")
        self.run_kmeans(wine_data_ica, y_wine, "wine_ica")
        self.run_kmeans(heart_data_ica, y_heart, "heart_ica")
        self.run_kmeans(wine_data_rp, y_wine, "wine_rp")
        self.run_kmeans(heart_data_rp, y_heart, "heart_rp")
        self.run_kmeans(wine_data_lda, y_wine, "wine_lda")
        self.run_kmeans(heart_data_lda, y_heart, "heart_lda")

    # ----------------- wine data ----------------
    #
    # plotx = []
    # ploty = []
    # plot_dict = {}
    #
    # plotx_ami = []
    # ploty_ami = []
    # plot_dict_ami = {}
    #
    # plotx_acc = []
    # ploty_acc = []
    # plot_dict_acc = {}
    #
    # adj_mi = defaultdict(lambda: defaultdict(float))
    #
    # print("Data head: \n", heart_data.head())
    #
    #
    # for n in n_clusters:
    #     k_means.set_params(n_clusters = n)
    #     k_means.fit(x_wine)
    #     labels = k_means.predict(x_wine)
    #
    #     silhouette_avg = silhouette_score(x_wine, labels)
    #     ami = adjusted_mutual_info_score(y_wine, labels)
    #     acc = accuracy_score(y_wine, labels)
    #     plotx.append(n)
    #     ploty.append(silhouette_avg)
    #
    #     plotx_ami.append(n)
    #     ploty_ami.append(ami)
    #
    #     plot_dict[n] = silhouette_avg
    #     plot_dict_ami[n] = ami
    #
    # plt.subplot(211)
    # plt.plot(plotx,ploty)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Avg. Silhouette Score')
    #
    # plt.subplot(212)
    # plt.plot(plotx_ami,ploty_ami)
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Adj. Mutual Info')
    #
    #
    # plt.savefig("kmeans_wine.png")
    # print("number of clusters with the highest avg. silhoutte score is : "+str(max(plot_dict.keys(), key=lambda k: plot_dict[k])))
    #
    # ----------------- heart data ----------------



    def run_kmeans(self, x_data, y_data, title):
        print(x_data.head())
        plt.figure()

        plotx = []
        ploty = []
        plot_dict = {}

        plotx_ami = []
        ploty_ami = []
        plot_dict_ami = {}

        adj_mi = defaultdict(lambda: defaultdict(float))

        # print("Data head: \n", heart_data.head())

        for n in self.n_clusters:
            self.k_means.set_params(n_clusters = n)
            self.k_means.fit(x_data)
            labels = self.k_means.predict(x_data)

            silhouette_avg = silhouette_score(x_data, labels)
            ami = adjusted_mutual_info_score(y_data, labels)
            plotx.append(n)
            ploty.append(silhouette_avg)

            plotx_ami.append(n)
            ploty_ami.append(ami)

            plot_dict[n] = silhouette_avg
            plot_dict_ami[n] = ami

        plt.figure()
        plt.subplot(211)
        plt.plot(plotx,ploty)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Avg. Silhouette Score')

        plt.subplot(212)
        plt.plot(plotx_ami,ploty_ami)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Adj. Mutual Info')

        plt.savefig("kmeans_" + title + ".png")


    # print("number of clusters with the highest avg. silhoutte score is : "+str(max(plot_dict.keys(), key=lambda k: plot_dict[k])))



kmeans = KMeansClustering()
kmeans.run_all_kmeans()

