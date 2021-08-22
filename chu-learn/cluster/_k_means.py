import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import utils

class KMeans:

    def __init__(self):
        self.common = utils.Common()


    def _initialize_representative_value(self, X: np.array) -> json:
        """
        초기화: 대표값으로 쓸 관측치 랜덤하게 k개 선택하여 dict로 반환
        {cluster_id: [rep_value1, rep_value2, ...]} 형식
        """
        n = self.n
        k = self.k

        sampled_idx = np.random.randint(n, size=k)
        rep_dic = {cluster_id: X[idx, :] for cluster_id, idx in enumerate(sampled_idx)}

        return rep_dic

    def _expectation(self, X: np.array, rep_dic: dict) -> np.array:
        """
        각 관측치들을 가까운 클러스터로 할당
        아웃풋 형태는 n*1의 np.array로 각 관측치의 클러스터를 나타내기
        """

        distances = []
        for cluster_id, cluster_rep in rep_dic.items():
            distance = ((X - cluster_rep) ** 2).sum(axis=1)
            distances.append(distance)

        distance_array = np.column_stack(distances)

        cluster_ids = distance_array.argmin(axis=1)

        return cluster_ids

    def _maximization(self, X: np.array, cluster_ids: np.array) -> dict:
        """
        각 클러스터별로 대푯값 재 생성
        """
        k = self.k
        new_rep_dic = {}
        for cluster_id in range(k):
            new_rep_dic[cluster_id] = X[cluster_ids == cluster_id].mean(axis=0)
        return new_rep_dic

    def _is_converged(self,
                      cluster_ids: np.array, new_cluster_ids: np.array,
                      rep_dic: dict, new_rep_dic: dict) -> bool:
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

    def fit(self,
            X: np.array, k: int, max_iter: int = 100,
            visualize_steps=False, verbose=0
            ):
        """
        학습 함수
        """
        self.k = k

        ## 초기화
        rep_dic = self._initialize_representative_value(X)
        cluster_ids = np.array([0] * len(X))

        ## 순회
        for iter_count in range(max_iter):

            new_cluster_ids = self._expectation(X, rep_dic)
            new_rep_dic = self._maximization(X, new_cluster_ids)

            ## 수렴 완료
            if self._is_converged(cluster_ids, new_cluster_ids, rep_dic, new_rep_dic):
                self.common._print_end_message(f"{iter_count}회 순회에서 수렴으로 알고리즘 종료")
                self.rep_dic = new_rep_dic
                self.cluster_ids = cluster_ids
                return

            ## 수렴 X
            else:
                rep_dic = new_rep_dic
                cluster_ids = new_cluster_ids

            if visualize_steps:
                self.visualize_cluster(X, rep_dic, cluster_ids)


        self.common._print_end_message(f"{iter_count}회 순회에서 수렴으로 알고리즘 종료")
        self.rep_dic = rep_dic

    def visualize_cluster(self, X, rep_dic, cluster_ids):
        """
        앞의 2개 칼럼만 시각화
        """
        color_list = ["b", "g", "r", "c", "m", "y", "k"]
        k = self.k

        for i in range(k):
            plt.scatter(rep_dic[i][0], rep_dic[i][1], s=1000, color=color_list[i], alpha=0.1)
            plt.scatter(X[cluster_ids == i, 0], X[cluster_ids == i, 1], color=color_list[i])
        plt.show()

    def predict(self, df: pd.DataFrame):
        """
        새롭게 주어진 data frame에 대해 클러스터 할당
        """
        pass