from matplotlib import pyplot as plt
import networkx as nx

# 初始化无向图
filename = "./facebook_combined.txt"
G = nx.Graph()
with open(filename) as file:
    for line in file:
        # 对输入的数据机型分割 格式为 0 2\n 取出尾部的\n 后依据 ' ' 进行分割得到两个结点
        str_edge = line.rstrip('\n').split(' ')
        # 将字符数据转化为整数数据
        int_edge = [int(x) for x in str_edge]
        # 将相应的边（无向）加入到无向图G中，自动将不在图中的结点加入
        G.add_edge(int_edge[0], int_edge[1])

print("Facebook社交圈无向图分析")
print("结点个数:{}".format(len(G.nodes)))
print("边个数:{}".format(len(G.edges)))

# 网络直径
diameter = 0
# # 计算平均路径长度
path_valve = []
for v in G.nodes:
    # 利用最短路径算法找出结点 v 与到其余各结点(u)的最短路径，注意包括结点 v 本身，路径为 0
    # 返回的结果为字典类型，为 u:length 的格式
    spl = nx.single_source_shortest_path_length(G, v)
    # 从字典中取出各个路径的长度加入到路径集合 path_valve 中
    for p in spl.values():
        path_valve.append(p)
        if p > diameter:
            diameter = p
# 对路径求和并除以路径的个数（由于计算了结点到自身的距离，故路径中多加了 G.nodes 个 0 需要减去）
# 对path_value 按升序排序 （注意剔除开头G.nodes个0）
path_valve.sort()
path_value_len = len(path_valve)
G_nodes_len = len(G.nodes)
# 计算平均路径长度
path_avg = sum(path_valve) / (path_value_len - G_nodes_len)
print("平均路径长度为:{}".format(path_avg))
# 收缩直径计算（即有效直径计算）有效直径：网络中90%的连通结点对可以相互到达的最短距离
# 可以采用将求得的所有最短路径进行从小到大排序，然后第 len(path_value)*90% 个路径就为有效路径
index = (path_value_len - G_nodes_len) * 0.9 + G_nodes_len
diameter_effective = path_valve[int(index)]
print("直径为:{}".format(diameter))
print("伸缩直径为:{}".format(diameter_effective))

# 计算聚集系数--------------------
# 存储所有的聚集系数
cluster_sum = 0
for v in G.nodes:
    # 计算每个结点的聚集系数并加入到 cluster_sum 中
    cluster_sum += nx.clustering(G, v)
# sum 除以结点个数得到平均聚集系数 cluster_avg
cluster_avg = cluster_sum / len(G.nodes)
print("平均聚集系数为:{}".format(cluster_avg))

# 度分布------------------------------
# 利用 degree_histogram 方法得到图 G 的分布序列，对应度从最小到最大出现的次数
# 例如 degree[2]=98 表示度为2的结点共有98个 sum(degree)=G.nodes
degree = nx.degree_histogram(G)
# 定义绘制度分布图的横、纵坐标
# 横坐标x 为0-len(degree) 的整数分布 即度出现次数由0 到 len(degree)-1
x = range(len(degree))
# 纵坐标y 为对应度出现的次数所占的比例，即度为x 的结点在全部结点中作战的比例
y = [z * 100 / float(sum(degree)) for z in degree]
# 将度分布图形化
plt.plot(x, y)
plt.title("Degree distribution", fontsize=20, color='r')
plt.xlabel('Degree', fontsize=20, color='k')
plt.ylabel('Percent', fontsize=20, color='k')
plt.show()
