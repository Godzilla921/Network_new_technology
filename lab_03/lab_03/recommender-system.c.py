import json
import os
import random
import time

import numpy as np

from tqdm import tqdm


class RecommenderSystem:
    def __init__(self, file_path, users, movies, ratings, sim_file, encoding_="ISO-8859-1"):
        # 初始化三个私有字段，表示三个文件的相对路径，用以读取文件中的内容
        self.__users_path = os.path.join(file_path, users)
        self.__movies_path = os.path.join(file_path, movies)
        self.__ratings_path = os.path.join(file_path, ratings)
        # 对文件中的数据进行预处理，并加载到相应集合中
        self.users_data = self.__load_user_data(encoding_)
        self.movies_data = self.__load_movie_data(encoding_)
        self.data = self.__load_data(encoding_)
        print("共计{}个样本".format(len(self.data)))
        # 测试集占0.1
        self.train_data, self.test_data = self.__splitData(0.1)
        print("训练集大小为:{}".format(len(self.train_data)))
        print("测试集大小为:{}".format(len(self.test_data)))
        # 初始化用户电影评价矩阵，行为userId, 列为movieId
        # 这是一个稀疏矩阵
        self.user_movie_matrix = self.__load_score_matrix()
        # 定义用户相似度矩阵
        self.user_cross_sim = self.__load_sim_matrix(os.path.join("./data", sim_file))

    # 加载用户，电影，评价的方法 格式[{userId,movieId,score},~~~]
    def __load_data(self, encoding_):
        temp_data = []
        with open(self.__ratings_path, 'r', encoding=encoding_) as f:
            for line in f.readlines():
                user_id, movie_id, score, _ = line.split("::")
                temp_data.append((user_id, movie_id, int(score)))
        return temp_data

    # 加载用户信息的方法 格式[{userId,gender,age,profession,email_address},~~~]
    def __load_user_data(self, encoding_):
        temp_data = []
        with open(self.__users_path, 'r', encoding=encoding_) as f:
            for line in f.readlines():
                user_id, gender, age, profession, email_address = line.rstrip().split("::")
                temp_data.append((user_id, gender, int(age), profession, email_address))
        return temp_data

    # 加载电影信息的方法 格式[{movieId, title, movie_class},~~~]
    def __load_movie_data(self, encoding_):
        temp_data = []
        with open(self.__movies_path, 'r', encoding=encoding_) as f:
            for line in f.readlines():
                movie_id, title, movie_class = line.rstrip().split("::")
                temp_data.append((movie_id, title, movie_class))
        return temp_data

    # 对样本数据进行切分,得到数据集和样本集
    # percent为 测试集 test所占的比例
    # 划分出训练集与测试集的方法
    def __splitData(self, percent):
        print("训练数据集与测试数据集切分...")
        train, test = [], []
        for user, movieId, score in self.data:
            # random.random() 随机产生 0-1的浮点数，小于percent的入测试集 大于的入训练集
            if random.random() < percent:
                test.append((user, movieId, score))
            else:
                train.append((user, movieId, score))
        return train, test

    def __load_score_matrix(self):
        # 用训练数据填充用户电影评价矩阵
        # 这是一个稀疏矩阵
        score_matrix = np.zeros((int(self.users_data[-1][0]),
                                 int(self.movies_data[-1][0])))
        for userId, movieId, score in self.train_data:
            score_matrix[int(userId) - 1][int(movieId) - 1] = score
        return score_matrix

    # 计算用户相似度矩阵
    def __load_sim_matrix(self, path):
        if os.path.exists(path):
            print("用户相似度从文件加载 ...")
            sim_matrix = np.array(json.load(open(path, "r")))
        else:
            print("开始计算用户之间的相似度 ...")
            sim_matrix = np.zeros((int(self.users_data[-1][0]),
                                   int(self.users_data[-1][0])))

            # ------------用于优化后的person相关系数矩阵计算的局部列表---
            # difference_matrix 二维列表存储每位用户对每部电影的打分-平均打分
            difference_matrix = self.__load_score_matrix()
            # average_score 一维列表存储每位用户的平均打分
            average_score = np.average(self.user_movie_matrix, axis=1)
            # 得到差值矩阵，即 x-ave_x
            for i in range(len(average_score)):
                difference_matrix[i] -= average_score[i]
            # sum_square 以为列表存储每位用户对所有电影（打分-平均打分）的平方和
            sum_square = np.sum(np.square(difference_matrix), axis=1)
            # 同一位用户的相似度为1
            for i in range(self.user_movie_matrix.shape[0]):
                sim_matrix[i, i] = 1
            # 对用用户评价矩阵的坐标轴0(0-6039)
            for i in tqdm(range(self.user_movie_matrix.shape[0])):
                # 对用用户评价矩阵的坐标轴0(0-6039)
                for j in range(i+1, self.user_movie_matrix.shape[0]):
                    # -----使用余弦相似度
                    # 计算用户i与用户j的余弦相似度
                    # sim_matrix[i, j] = self.sim_cosine(self.user_movie_matrix[i, :],
                    #                                    self.user_movie_matrix[j, :])

                    # -----使用皮尔逊相关系数
                    # 直接使用sim_person 函数会有大量的重复计算，性能较差
                    # sim_matrix[i, j] = self.sim_person(self.user_movie_matrix[i, :],
                    #                                    self.user_movie_matrix[j, :])

                    # ---------------- 优化后的person相关系数计算----------------------
                    # 一些重复的计算依据已预先计算并存储，在这里可以直接调用
                    sim_matrix[i, j] = np.sum(difference_matrix[i, :] * difference_matrix[j, :]) \
                                       / np.sqrt(sum_square[i] * sum_square[j])
                    # 对user_cross_sim矩阵进行填充，得到对称矩阵
                    sim_matrix[j, i] = sim_matrix[i, j]
            json.dump(sim_matrix.tolist(), open(path, 'w'))
        return sim_matrix

    # 根据用户相似度，预测此用户对此电影的打分
    def predicting_score(self, user_id, movie_id):
        # 利用 numpy 进行计算提高程序性能
        # index 为对该电影评分不为0 的用户
        index = np.nonzero(self.user_movie_matrix[:, movie_id])
        # sum_w 为所有其他看过这部电影的用户与该用户的相似度之和
        sum_w = np.sum(self.user_cross_sim[user_id, index])
        # sum_w_rating 为依据与每个用户的相似度计算出的分数之和
        sum_w_rating = np.sum(self.user_cross_sim[user_id, index] * self.user_movie_matrix[index, movie_id])
        # 由于缺失几部电影，所有用户都未看过缺失的电影，故直接预测该用户不会看该电影
        if sum_w == 0.0:
            return 0.0
        else:
            return sum_w_rating / sum_w

    def precision(self):
        print("正在计算准确率 ...")
        time.sleep(0.1)
        # 对于每一个训练样本
        # score 为实际分数，score_predicting为预测分数
        # predicting_success统计预测成功的数目
        predicting_success = 0
        for user_id, movie_id, score in tqdm(self.test_data):
            # 得到预测分数
            score_predicting = self.predicting_score(int(user_id) - 1, int(movie_id) - 1)
            # 按照四舍五入进行取整
            score_predicting = int(score_predicting + 0.5)
            if int(score) == int(score_predicting):
                predicting_success += 1
        # 准确度等于 成功预测的数目/总体测试数据数目
        return float(predicting_success) / len(self.test_data)

    # 计算相似度函数
    # 余弦相似度
    @staticmethod
    def sim_cosine(x, y):
        # 若向量x 或向量y 的二范数为0 直接返回0
        if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
            return 0
        # 返回向量x 与向量y 的余弦值
        # 计算公式 向量x 与 y 的内积除以x与y的二范数乘积
        return np.sum(x * y) / (np.linalg.norm(x) * np.linalg.norm(y))

    # 计算皮尔逊相关系数
    @staticmethod
    def sim_person(x, y):
        # 先计算得到向量x与向量y的平均值ave_x，ave_y
        ave_x = np.average(x)
        ave_y = np.average(y)
        # 计算皮尔森相关基数 sigma 为求和函数
        # 公式为sim(x,y)=sigma((x-ave_x)*(y-ave_y))/(sqrt(sigma((x-ave_x)^2)*sigma((y-ave_y)^2)))
        return np.sum((x - ave_x) * (y - ave_y)) / np.sqrt(
            (np.sum((x - ave_x) * (x - ave_x)) * np.sum((y - ave_y) * (y - ave_y))))


my_recommender_system = RecommenderSystem("./ml-1m", "users.dat", "movies.dat", "ratings.dat",
                                          "pearson_sim_version_2.json")
print("预测准确度为:{}".format(my_recommender_system.precision()))
