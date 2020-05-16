import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# LOAD MAT DATA
def load_data():
    print('LOADING DATA...')
    mat = sio.loadmat('./corel.mat')
    corel_features = mat['corel_features']
    corel_labels = mat['corel_labels']
    return corel_features, corel_labels

# K-MEANS CLUSTER
def kmeans_cluster(corel_features, corel_labels, k_final):
    print('K-MEANS CLUSERING...')
    corel_df = pd.DataFrame(data = corel_features)
    pred_acc_list = []
    pred_sse_list = []
    for k in range(1, k_final + 1):
        corel_kmeans = KMeans(n_clusters = k).fit(corel_df)
        pred_centroids = corel_kmeans.cluster_centers_
        pred_labels = corel_kmeans.labels_
        # ACCURACY (USELESS FOR THIS EXERCISE)
        pred_acc = metrics.accuracy_score(corel_labels, pred_labels)
        pred_acc_list.append(pred_acc)
        # SUM OF SQUARED DISTANCE
        pred_sse = corel_kmeans.inertia_
        pred_sse_list.append(pred_sse)
    return pred_acc_list, pred_sse_list

# K-MEANS PLOT
def kmeans_plot(pred_acc_list, pred_sse_list, k_final):
    print('K-MEANS PLOTTING...')
    k = range(1, k_final + 1)
    # plt.plot(k, pred_acc_list)
    plt.plot(k, pred_sse_list)
    plt.title('K-Means Sum of Squared Distance')
    plt.xlabel('Number of Centroids')
    plt.ylabel('Sum of Square Distance')
    plt.show()

# PCA TO REDUCE DIMENSIONS
def run_pca(corel_features):
    print('PCA RUNNING...')
    corel_df = pd.DataFrame(data = corel_features)
    x = corel_df.values
    pca_data = PCA(n_components = 2).fit_transform(x)
    pca_df = pd.DataFrame(data = pca_data)
    return pca_data

# PCA PLOT PROJECTION WITH LABELS
def pca_plot(pca_data, corel_labels):
    print('PCA PROJECTION PLOTTING...')
    corel_labels = np.reshape(corel_labels, -1)
    plt.scatter(pca_data[:, 0], pca_data[:, 1],
                c = corel_labels, edgecolor = 'none', alpha = 0.5,
                cmap = plt.cm.get_cmap('Accent', 18))
    plt.title('PCA Projection Plot')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.show()

# FIND SUITABLE PCA
def find_pca_dim(corel_features, max_dim):
    print('PCA SUITABLE DIMS FINDING...')
    corel_df = pd.DataFrame(data = corel_features)
    x = corel_df.values
    reconstruction_error_list = []
    for d in range(1, max_dim + 1):
        pca = PCA(n_components = d)
        pca_data = pca.fit_transform(x)
        project_back_data = pca.inverse_transform(pca_data)
        error = LA.norm((x - project_back_data), None)
        reconstruction_error_list.append(error)
    return reconstruction_error_list

# RECONSTRUCTION ERROR
def pca_error_plot(reconstruction_error_list, max_dim):
    print('RECONSTRUCTION ERROR PLOTING...')
    d = range(1, max_dim + 1)
    plt.plot(d, reconstruction_error_list)
    plt.title('PCA Reconstruction Error')
    plt.xlabel('Number of Component')
    plt.ylabel('Reconstruction Error')
    plt.show()


# K-MEANS AND CLUSTER COMBINATION
def kmeans_pca(corel_features, corel_labels, k):
    print('K-MEANS CLUSERING...')
    corel_df = pd.DataFrame(data = corel_features)
    x = corel_df.values
    pca_data = PCA(n_components = 20).fit_transform(x)
    pca_data_df = pd.DataFrame(data = pca_data)
    corel_kmeans = KMeans(n_clusters = k).fit(pca_data_df)
    pred_centroids = corel_kmeans.cluster_centers_
    pred_labels = corel_kmeans.labels_
    print('PCA RUNNING...')
    corel_df = pd.DataFrame(data = corel_features)
    x = corel_df.values
    pca_data = PCA(n_components = 2).fit_transform(x)
    pca_centroid = PCA(n_components = 2).fit_transform(pred_centroids)
    print('PCA PLOTING...')
    pred_labels = np.reshape(pred_labels, -1)
    plt.scatter(pca_data[:, 0], pca_data[:, 1],
                c = pred_labels, edgecolor = 'none', alpha = 0.5,
                cmap = plt.cm.get_cmap('Accent', k))
    plt.title('PCA Projection Plot')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    plt.scatter(pca_centroid[:, 0], pca_centroid[:, 1], c = 'black', label = 'centroids')
    plt.show()


# # K-MEANS AND CLUSTER COMBINATION
# def kmeans_pca(corel_features, corel_labels, k):
#     print('K-MEANS CLUSERING...')
#     corel_df = pd.DataFrame(data = corel_features)
#     x = corel_df.values
#     pca_data = PCA(n_components = 20).fit_transform(x)
#     pca_data_df = pd.DataFrame(data = pca_data)
#     corel_kmeans = KMeans(n_clusters = k).fit(pca_data_df)
#     pred_centroids = corel_kmeans.cluster_centers_
#     pred_labels = corel_kmeans.labels_
#     print('PCA RUNNING...')
#     corel_df = pd.DataFrame(data = corel_features)
#     x = corel_df.values
#     pca_data = PCA(n_components = 2).fit_transform(x)
#     pca_centroid = PCA(n_components = 2).fit_transform(pred_centroids)
#     print('PCA PLOTING...')
#     pred_labels = np.reshape(pred_labels, -1)
#     plt.scatter(pca_data[:, 0], pca_data[:, 1],
#                 c = pred_labels, edgecolor = 'none', alpha = 0.5,
#                 cmap = plt.cm.get_cmap('Accent', k))
#     plt.title('PCA Projection Plot')
#     plt.xlabel('Component 1')
#     plt.ylabel('Component 2')
#     plt.colorbar()
#     plt.scatter(pca_centroid[:, 0], pca_centroid[:, 1], c = 'black', label = 'centroids')
#     plt.show()

if __name__ == "__main__":
    # LOAD DATA
    corel_features, corel_labels = load_data()

    # # K_MEANS CLUSTER PART
    # k_final = 50
    # pred_acc_list, pred_sse_list = kmeans_cluster(corel_features, corel_labels, k_final)
    # kmeans_plot(pred_acc_list, pred_sse_list, k_final)

    # # PCA ANALYSIS PART
    # pca_data = run_pca(corel_features)
    # pca_plot(pca_data, corel_labels)

    # # PCA DIMS SELECTION PART
    # max_dim = 144
    # reconstruction_error_list = find_pca_dim(corel_features, max_dim)
    # pca_error_plot(reconstruction_error_list, max_dim)

    # K-MEANS AND PCA PLOT
    k = 10
    kmeans_pca(corel_features, corel_labels, k)