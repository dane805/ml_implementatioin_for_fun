import json

from scipy.stats import multivariate_normal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .. import cluster


class GaussianMixture:

    def __init__(self):
        self.common = utils.Common()

    def _initialize_representative_value(self, X: np.array) -> json:
        """
        초기화: k means로 초깃 클러스터 부여
        """
        k = self.k
        kmeans = cluster.KMeans()
        kmeans.fit(X, k, max_iter=1)
        cluster_ids = kmeans.cluster_ids
        param_dic = self._estimate_parameters_mle(X, cluster_ids)

        self.cluster_ids = cluster_ids
        self.param_dic = param_dic

    def _expectation(self, X: np.array, rep_dic: dict) -> np.array:
        """
        각 관측치들을 가까운 클러스터로 할당
        아웃풋 형태는 n*1의 np.array로 각 관측치의 클러스터를 나타내기
        """
        n = self.n
        n_features = self.n_features
        means = rep_dic['means']
        covariances = rep_dic['covariances']

        likelihoods = np.empty((n, n_features))
        for i in range(k):
            cluster_mean = means[i]
            covariance = covariances[i]
            rv = multivariate_normal(cluster_mean, covariance)
            likelihood = np.log(rv.pdf(X))
            likelihoods[:, i] = likelihood

        cluster_ids = likelihoods.argmax(axis=1)

        return cluster_ids

    def _estimate_parameters_mle(self, X, cluster_ids):
        """
        MLE 베이스로 정규 분포 parameter 추정
        """
        k = self.k
        n_features = self.n_

        means = np.empty(k)
        covariances = np.empty((k, n_features, n_features))
        for i in range(k):
            X_cluster = X[cluster_ids == i, :]

            cluster_mean = np.mean(X_cluster)

            deviance = X_cluster - cluster_mean
            covariance = deviance.T @ deviance / len(X_cluster)

            means[i] = cluster_mean
            covariances[i] = covariance
        param_dic = {
            "means" : means,
            "covariances" : covariances
        }
        return param_dic

    def _maximization(self, X: np.array, cluster_ids: np.array) -> dict:
        """
        각 클러스터별로 대푯값 재 생성
        공분산 행렬은 왜 자유도 안 빼주지 ??
        """

        means, covariances = self._estimate_parameters_mle(X)

        new_params = {
            "means": means,
            "covariances": covariances
        }

        return new_params

    def _is_converged(self,
                      cluster_ids: np.array, new_cluster_ids: np.array,
                      parameters: dict, new_parameters: dict) -> bool:
        """
        알고리즘이 수렴했는지 판단
        가장 쉽게, 변하지 않았으면 수렴으로 판단
        """

        for cluster_id in rep_dic:
            if np.any(rep_dic[cluster_id] != new_rep_dic[cluster_id]):
                return False

        if np.any(new_cluster_ids != cluster_ids):
            return False

        return True

    def fit(self, X: np.array, k: int, max_iter: int = 100, visualize_steps=False):
        """
        학습 함수
        """
        self.k = k
        self.n, self.n_features = X.shape

        n = self.n
        n_features = self.n_features

        ## 초기화
        rep_dic = self._initialize_representative_value(X)
        cluster_ids = np.array([0] * len(X))

        ## 순회
        for iter_count in range(max_iter):

            cluster_onehot = self._expectation(X, rep_dic)
            new_rep_dic = self._maximization(X, new_cluster_ids)

            ## 수렴 완료
            if self._is_converged(cluster_ids, new_cluster_ids, rep_dic, new_rep_dic):
                print(f"{iter_count}회 순회에서 수렴으로 알고리즘 종료")
                self.rep_dic = new_rep_dic
                self.cluster_ids = cluster_ids
                return

            ## 수렴 X
            else:
                rep_dic = new_rep_dic
                cluster_ids = new_cluster_ids

            if visualize_steps:
                self.visualize_cluster(rep_dic, cluster_ids)

            print(rep_dic)

        print("최대 순회 횟수 초과로 알고리즘 종료")
        self.rep_dic = rep_dic
        self.cluster_ids = cluster_ids

    def visualize_cluster(self, rep_dic, cluster_ids):
        """
        앞의 2개 칼럼만 시각화
        """
        color_list = ["b", "g", "r", "c", "m", "y", "k"]

        leg = []
        for i in range(k):
            plt.scatter(rep_dic[i][0], rep_dic[i][1], s=1000, color=color_list[i], alpha=0.1)
            plt.scatter(X[cluster_ids == i, 0], X[cluster_ids == i, 1], color=color_list[i])
        plt.show()

    def predict(self, df: pd.DataFrame):
        """
        새롭게 주어진 data frame에 대해 클러스터 할당
        """
        if not self.rep_dic:
            raise Exception("학습되지 않아 예측할 수 없습니다")