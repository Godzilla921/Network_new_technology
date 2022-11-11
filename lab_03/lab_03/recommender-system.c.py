import json
import os
import random
import numpy as np

from tqdm import tqdm


class RecommenderSystem:
    def __init__(self, file_path, users, movies, ratings, encoding_="ISO-8859-1"):
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
        self.user_cross_sim = self.__load_sim_matrix(os.path.join("./data", "pearson_sim.json"))

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
            # 对用用户评价矩阵的坐标轴0(0-6039)
            for i in tqdm(range(self.user_movie_matrix.shape[0])):
                # 对用用户评价矩阵的坐标轴0(0-6039)
                for j in range(i, self.user_movie_matrix.shape[0]):
                    # 若是同一位用户，相似度为1
                    if i == j:
                        sim_matrix[i, j] = 1
                    else:
                        # 使用余弦相似度
                        # 计算用户i与用户j的余弦相似度
                        # sim_matrix[i, j] = self.sim_cosine(self.user_movie_matrix[i, :],
                        #                                    self.user_movie_matrix[j, :])
                        # 使用皮尔逊相关系数
                        sim_matrix[i, j] = self.sim_person(self.user_movie_matrix[i, :],
                                                           self.user_movie_matrix[j, :])
                    # 对user_cross_sim矩阵进行填充，得到对称矩阵
                    sim_matrix[j, i] = sim_matrix[i, j]
            json.dump(sim_matrix.tolist(), open(path, 'w'))
        return sim_matrix

    # 根据用户相似度，预测此用户对此电影的打分
    def predicting_score(self, user_id, movie_id):
        sum_w = 0.0         # sum_w 为所有其他看过这部电影的用户与该用户的相似度之和
        sum_w_rating = 0.0  # sum_w_rating 为依据与每个用户的相似度计算出的分数之和
        for i in range(6040):
            # 用户i 看过电影 movie_id
            if self.user_movie_matrix[i, movie_id] != 0:
                # swm_w 为用户相似度之和
                sum_w += self.user_cross_sim[user_id, i]
                #
                sum_w_rating += self.user_cross_sim[user_id, i] * self.user_movie_matrix[i, movie_id]
        if sum_w == 0.0:
            return 0.0
        else:
            return sum_w_rating / sum_w

    def precision(self):
        print("正在计算准确率 ...")
        # 对于每一个训练样本
        # score 为实际分数，score_predicting为预测分数
        # predicting_success统计预测成功的数目
        predicting_success = 0
        for user_id, movie_id, score in tqdm(self.test_data):
            # 得到预测分数，平按照四舍五入进行取整
            score_predicting = self.predicting_score(int(user_id) - 1, int(movie_id) - 1)
            score_predicting = int(score_predicting + 0.5)
            if score == score_predicting:
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


my_recommender_system = RecommenderSystem("./ml-1m", "users.dat", "movies.dat", "ratings.dat")
print("预测准确度为:{}".format(my_recommender_system.precision()))
