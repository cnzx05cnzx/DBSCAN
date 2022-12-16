import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import random

UNVISITED = -2
# 噪声
NOISE = -1


def seed_torch(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def dist(a, b):
    return math.sqrt(np.power(a - b, 2).sum())


def whether_neighbor(a, b, radius):
    return dist(a, b) < radius


def whether_kernel_object(dataSet, selectId, radius):
    """
    输入：数据集, 查询是否是核心对象的点id, 半径大小ϵ
    输出：[ 在领域范围内的点的id ]
    """
    # 计算在领域范围内有多少点
    return [i for i in range(dataSet.shape[0]) if whether_neighbor(a=dataSet[selectId], b=dataSet[i], radius=radius)]


def get_data(path):
    data = pd.read_csv(path)[:1000]
    columns = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT',
               'PAYMENTS', 'MINIMUM_PAYMENTS']

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
    return data


def DB_SCAN(dataSet, radius=4.0, minPts=10):
    pointNumber = dataSet.shape[0]
    # 和伪代码一致先将所有的点标为未访问，flagClusterList最终是存放每个点的判定结果，例如簇几、噪声
    flagClusterList = [UNVISITED] * pointNumber
    # 簇的数目，也算作下标，从簇1开始
    # (当前的簇)
    clusterNumber = 0
    # 开始遍历
    for selectPoint in range(pointNumber):
        # 如果点没有被分类
        if flagClusterList[selectPoint] == UNVISITED:
            # 找出邻域里的点
            neighbors = whether_kernel_object(dataSet=dataSet, selectId=selectPoint, radius=radius)
            # 不满足最小点
            if len(neighbors) < minPts:
                flagClusterList[selectPoint] = NOISE
            else:
                # 划分到当前簇
                flagClusterList[selectPoint] = clusterNumber
                # 先将邻域里的其他点装到簇里（和伪代码一致）
                for neighborsId in neighbors:
                    # 未访问过或是噪声(若是其他簇的话按照先来后到的顺序)
                    if flagClusterList[neighborsId] == UNVISITED or flagClusterList[neighborsId] == NOISE:
                        flagClusterList[neighborsId] = clusterNumber
                # 扩张
                while len(neighbors) > 0:
                    # 掏出第一个点
                    currentPoint = neighbors[0]
                    # 查询当前点邻域中的点
                    queryNeighbors = whether_kernel_object(dataSet=dataSet, selectId=currentPoint, radius=radius)
                    # 如果大于等于最小点
                    if len(queryNeighbors) >= minPts:
                        for i in range(len(queryNeighbors)):
                            # 将新的邻域点取出存到N中
                            resultPoint = queryNeighbors[i]
                            if flagClusterList[resultPoint] == UNVISITED:
                                # 如果未访问的话继续往下找，继续扩展
                                neighbors.append(resultPoint)
                                flagClusterList[resultPoint] = clusterNumber
                            # 和上面思想一致
                            elif flagClusterList[resultPoint] == NOISE:
                                flagClusterList[resultPoint] = clusterNumber
                    # 因为N中第一个拿过了，需要刷新
                    neighbors = neighbors[1:]
                # 该簇结束，创建一个新簇
                clusterNumber += 1
    return flagClusterList, clusterNumber


if __name__ == '__main__':
    seed_torch()
    dataSet = get_data("card.csv")
    X = np.asarray(dataSet)
    scale = StandardScaler()
    X = scale.fit_transform(X)
    tsne = TSNE(n_components=2, init="random", learning_rate="auto")
    dataSet = tsne.fit_transform(X)

    print('使用自己db-scan')
    label_list, clusterNumber = DB_SCAN(dataSet, radius=3.5, minPts=10)
    p1 = sns.scatterplot(data=dataSet, x=dataSet[:, 0], y=dataSet[:, 1], hue=label_list, legend="full", palette="deep")
    sns.move_legend(p1, "upper right", bbox_to_anchor=(1.15, 1.1), title='Clusters')
    plt.show()

    # print('使用库中db-scan')
    # clusters = DBSCAN(eps=3.5, min_samples=10).fit(dataSet)
    # p2 = sns.scatterplot(data=dataSet, x=dataSet[:, 0], y=dataSet[:, 1], hue=clusters.labels_, legend="full",
    #                      palette="deep")
    # sns.move_legend(p2, "upper right", bbox_to_anchor=(1.15, 1.1), title='Clusters')
    # plt.show()
