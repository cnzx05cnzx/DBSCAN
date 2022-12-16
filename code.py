import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# 数据读取
if __name__ == '__main__':
    data = pd.read_csv("card.csv")[:1000]
    columns = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
               'PAYMENTS', 'MINIMUM_PAYMENTS']

    # 数据预处理
    for c in columns:
        Range = c + '_RANGE'
        data[Range] = 0
        data.loc[((data[c] > 0) & (data[c] <= 500)), Range] = 1
        data.loc[((data[c] > 500) & (data[c] <= 1000)), Range] = 2
        data.loc[((data[c] > 1000) & (data[c] <= 3000)), Range] = 3
        data.loc[((data[c] > 3000) & (data[c] <= 5000)), Range] = 4
        data.loc[((data[c] > 5000) & (data[c] <= 10000)), Range] = 5
        data.loc[((data[c] > 10000)), Range] = 6

    columns = ['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
               'PURCHASES_INSTALLMENTS_FREQUENCY',
               'CASH_ADVANCE_FREQUENCY', 'PRC_FULL_PAYMENT']

    for c in columns:
        Range = c + '_RANGE'
        data[Range] = 0
        data.loc[((data[c] > 0) & (data[c] <= 0.1)), Range] = 1
        data.loc[((data[c] > 0.1) & (data[c] <= 0.2)), Range] = 2
        data.loc[((data[c] > 0.2) & (data[c] <= 0.3)), Range] = 3
        data.loc[((data[c] > 0.3) & (data[c] <= 0.4)), Range] = 4
        data.loc[((data[c] > 0.4) & (data[c] <= 0.5)), Range] = 5
        data.loc[((data[c] > 0.5) & (data[c] <= 0.6)), Range] = 6
        data.loc[((data[c] > 0.6) & (data[c] <= 0.7)), Range] = 7
        data.loc[((data[c] > 0.7) & (data[c] <= 0.8)), Range] = 8
        data.loc[((data[c] > 0.8) & (data[c] <= 0.9)), Range] = 9
        data.loc[((data[c] > 0.9) & (data[c] <= 1.0)), Range] = 10
    data.drop(['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
               'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
               'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
               'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
               'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
               'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT'], axis=1, inplace=True)

    # 数据标准化
    X = np.asarray(data)
    scale = StandardScaler()
    X = scale.fit_transform(X)

    # 使用k邻寻找dbscan的最佳参数
    nbrs = NearestNeighbors(n_neighbors=6).fit(X)
    neigh_dist, neigh_ind = nbrs.kneighbors(X)
    sort_neigh_dist = np.sort(neigh_dist, axis=0)
    k_dist = sort_neigh_dist[:, 4]
    plt.plot(k_dist)
    plt.axhline(y=3.5, linewidth=1, linestyle='dashed', color='k')
    plt.ylabel("k-NN distance")
    plt.xlabel("Sorted observations (6th NN)")
    plt.show()

    # t-sne降维
    tsne = TSNE(n_components=2)
    Y = tsne.fit_transform(X)

    # dbscan聚类
    clusters = DBSCAN(eps=3.5, min_samples=10).fit(Y)

    # 聚类结果可视化
    data['t-SNE-1'] = Y[:, 0]
    data['t-SNE-2'] = Y[:, 1]
    p = sns.scatterplot(data=data, x="t-SNE-1", y="t-SNE-2", hue=clusters.labels_, legend="full", palette="deep")
    sns.move_legend(p, "upper right", bbox_to_anchor=(1.15, 1.1), title='Clusters')
    plt.show()
