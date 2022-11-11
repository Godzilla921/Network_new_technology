import math
import random


class UserCFRec:
    def __init__(self, datafile):
        self.datafile = datafile  # 初始化文件路径
        self.data = self.loadData()
        print("共计{}个样本".format(len(self.data)))
        # 训练集与数据集
        # 格式为 {userId:{movieId:score,~~~},~~~}
        self.trainData, self.testData = self.splitData(0.1)
        self.users_sim = self.UserSimilarityBest()

    # 对数据样本进行预处理，得到data列表，表中每一项为一个元组表示 (userId,movieId,score)
    # 这是全部数据
    def loadData(self):
        print("加载数据...")
        data = []
        for line in open(self.datafile):
            userid, itemid, score, _ = line.split(",")
            data.append((userid, itemid, float(score)))
        return data

    # 对样本数据进行切分,得到数据集和样本集
    # percent为 测试集 test所占的比例
    def splitData(self, percent):
        print("训练数据集与测试数据集切分...")
        train, test = {}, {}
        # random.seed(seed) # 设置随机数种子
        for user, item, record in self.data:
            # random.random() 随机产生 0-1的浮点数，小于percent的入测试集 大于的入训练集
            if random.random() < percent:
                test.setdefault(user, {})
                test[user][item] = record
            else:
                train.setdefault(user, {})
                train[user][item] = record
        return train, test

    def UserSimilarityBest(self):
        print("开始计算用户之间的相似度 ...")
        # 得到每个item被哪些user评价过
        # 格式为 {movieId:{userId,~~~~},~~~~}
        movie_users = dict()
        for userId, movieId_scores in self.trainData.items():
            for movieId in movieId_scores.keys():
                movie_users.setdefault(movieId, set())
                # 用户对该电影有评分，即看过该电影
                if self.trainData[userId][movieId] > 0:
                    # 将该用户加到该电影所对应的用户Id集合中
                    movie_users[movieId].add(userId)
        # 得到电影到用户的倒查表movie_users，表示哪些用户看过该电影
        # --------------------------

        # 根据倒查表，建立用户相似度矩阵
        count = dict()

        # user_item_count 记录用户看过的电影总数量，即集合的大小 格式{userId:count,~~~~}
        user_item_count = dict()
        # 取出倒查表中的电影ID与看过该电影的用户ID集合users
        for movieId, users in movie_users.items():
            # 对每一位看过该电影的用户
            for userId1 in users:
                # 若字典中无userId1 ，则加入，并将count初始化为0
                user_item_count.setdefault(userId1, 0)
                # userId1 看过的电影数目将+1
                user_item_count[userId1] += 1
                # 若字典count中无，userId1,则加入，并将value初始化为空字典{}
                count.setdefault(userId1, {})
                # 对于用户对（userId1,userId2），二者均看过电影movieId
                for userId2 in users:
                    # 先将用户对value初始化为0
                    count[userId1].setdefault(userId2, 0)
                    if userId1 == userId2:  # 取出自己与自己的用户对
                        continue
                    # 利用count记录userId1与userId2看过的相同电影的数目
                    # 用户对（userId1,userId2）共同看过的电影数目+1
                    count[userId1][userId2] += 1
                # 得到用户看过相同电影数目的矩阵 user_item_count
                # ----------------------------

        # 构建相似度矩阵 userSim
        userSim = dict()
        for userId1, related_users in count.items():
            userSim.setdefault(userId1, {})
            for userId2, cuv in related_users.items():
                # 对于用户对(userId1,userId2)
                if userId1 == userId2: # 若相同则直接跳过
                    continue
                # 若不存在，则加入，并将value初始化为0
                userSim[userId1].setdefault(userId2, 0.0)
                # 计算用户对 (userId1,userId2) 的相似度
                # 公式为二者看过相同电影的数目/sqrt(userID1看过的电影数目*userID2看过的电影数目)，即余弦计算相似·
                userSim[userId1][userId2] = cuv / math.sqrt(user_item_count[userId1] * user_item_count[userId2])
        return userSim

    def recommend(self, user, k=8, nitems=40):
        result = dict()
        have_score_items = self.trainData.get(user, {})
        for v, wuv in sorted(self.users_sim[user].items(), key=lambda x: x[1], reverse=True)[0:k]:
            for i, rvi in self.trainData[v].items():
                if i in have_score_items:
                    continue
                result.setdefault(i, 0)
                result[i] += wuv * rvi
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[0:nitems])

    def precision(self, k=8, nitems=10):
        print("开始计算准确率 ...")
        hit = 0
        precision = 0
        for user in self.trainData.keys():  # 对于训练集中的每位用户
            tu = self.testData.get(user, {})  # 我们得到测试集该用户的电影喜好程度
            rank = self.recommend(user, k=k, nitems=nitems)
            for item, rate in rank.items():
                if item in tu:
                    hit += 1
            precision += nitems
        return hit / (precision * 1.0)


userCFRec = UserCFRec("./ml-1m/ratings.csv")
# userCFRec = UserCFRec("./ml-1m/ratings.dat")
print(userCFRec.precision())