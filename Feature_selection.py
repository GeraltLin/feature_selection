# encoding: utf-8
'''
@author: linwenxing
@contact: linwenxing@zbj.com
'''

from sklearn.feature_selection import VarianceThreshold  # 导入python的相关模块
import matplotlib.pylab as plt
from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

X= pd.DataFrame([[0, 0, 1,1], [0, 1, 0,1], [1, 0, 0,1], [0, 1, 1,1], [0, 1, 0,1], [0, 1, 1,1]])  # 其中包含6个样本，每个样本包含3个特征。
# print(X.)
#
# # print(X)
# sel = VarianceThreshold(threshold=(0.8 * (1 - 0.8)))  # 表示剔除特征的方差大于阈值的特征Removing features with low variance
# sel.fit_transform(X)  # 返回的结果为选择的特征矩阵
# #
# # print(sel.fit_transform(X))  #
# print(sel.variances_)
#
from sklearn.datasets import load_iris
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
# # X.shape
# # (150, 4)
# X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
# # X_new.shape
#
# print(X_new)


class FeatureSelection():

    def __init__(self,X,y):
        self.X = X
        self.y = y
        self.variance_sel_feature = None
        self.chi_sel_feature = None

        if isinstance(X,DataFrame):
            self.feature_name = list(map(lambda x:str(x),X.columns.values.tolist()))
        else:
            clos = np.shape(X)[1]
            self.feature_name = list(map(lambda x:str(x+1),range(clos)))

    def varianceSelction(self,threshold):
        """
        根据方差进行特征选择
        :param threshold:方差选择的阈值
        :return:
        """
        variance_sel = VarianceThreshold(threshold=threshold)
        self.variance_sel_feature = variance_sel.fit_transform(self.X)
        variance =  variance_sel.variances_
        return variance


    def chiSelction(self,k='all'):
        """
        根据卡方检验进行特征选择
        :param k: 包留的特征数量
        :return:
        """

        chi_sel = SelectKBest(chi2, k=k)  # 选择k个最佳特征
        self.chi_sel_feature = chi_sel.fit_transform(self.X, self.y )  # ir
        score = chi_sel.scores_
        return score


    def treeSelection(self):
        """
        根据树模型来选择特征
        :return:
        """

        tc_sel = ExtraTreesClassifier()
        tc_sel = tc_sel.fit(self.X,self.y)
        model = SelectFromModel(tc_sel,prefit=True)
        self.tc_sel_feature = model.transform(self.X)
        score = tc_sel.feature_importances_
        return score



    def matplot_metrics(self,metrics,topk=None):
        """
        可视化特征的得分
        :param metrics:
        :param topk:
        :return:
        """

        if isinstance(metrics[0], float):
            metrics = list(map(lambda x: round(x, 2),metrics))

        plt.rcParams['font.sans-serif'] = ['SimHei']
        data = list(zip(self.feature_name,metrics))
        data = sorted(data,key=lambda x:x[1])
        y = list(map(lambda x:x[0],data))
        x = list(map(lambda x:x[1],data))

        if topk:
            y = y[:topk]
            x = x[:topk]
        fig, ax = plt.subplots()
        b = ax.barh(range(len(y)), x, color='#6699CC')

        for rect in b:
            w = rect.get_width()
            ax.text(w, rect.get_y() + rect.get_height() / 2, '%s' %(w), ha='left', va='center')
        ax.set_yticks(range(len(y)))
        ax.set_yticklabels(y)
        plt.barh(y, x)
        plt.ylabel('特征')
        plt.xlabel('指标')
        plt.show()




if __name__ == '__main__':
    FS = FeatureSelection(X, y)

    # variance = FS.varianceSelction(0.8 * (1 - 0.8))
    score = FS.chiSelction()
    score = FS.treeSelection()

    FS.matplot_metrics(score)
    # print(FS.variance_sel_feature)



