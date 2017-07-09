#coding:utf-8
#迪杰斯特拉算法:求无向图的最短路径，权值为正
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Dijkstra算法——通过边实现松弛
# 指定一个点到其他各顶点的路径——单源最短路径


# 每次找到离源点最近的一个顶点，然后以该顶点为重心进行扩展
# 最终的到源点到其余所有点的最短路径
# 一种贪婪算法
def Dijkstra(G, v0, INF=999):
    """ 使用 Dijkstra 算法计算指定点 v0 到图 G 中任意点的最短路径的距离
        INF 为设定的无限远距离值
        此方法不能解决负权值边的图
    """
    book = set()
    minv = v0

    # 源顶点到其余各顶点的初始路程
    dis = dict((k, INF) for k in G.keys())
    dis[v0] = 0

    while len(book) < len(G):
        book.add(minv)  # 确定当期顶点的距离
        for w in G[minv]:  # 以当前点的中心向外扩散
            if dis[minv] + G[minv][w] < dis[w]:  # 如果从当前点扩展到某一点的距离小与已知最短距离
                dis[w] = dis[minv] + G[minv][w]  # 对已知距离进行更新

        new = INF  # 从剩下的未确定点中选择最小距离点作为新的扩散点
        for v in dis.keys():
            if v in book: continue
            if dis[v] < new:
                new = dis[v]
                minv = v
    return dis


# # 初始化图参数:无向图邻接表
G = {
    1: {
        1: 0,
        2: 1,
        3: 12
    },
    2: {
        2: 0,
        3: 9,
        4: 3
    },
    3: {
        3: 0,
        5: 5
    },
    4: {
        3: 4,
        4: 0,
        5: 13,
        6: 15
    },
    5: {
        5: 0,
        6: 4
    },
    6: {
        6: 0
    }
}

graph = nx.Graph()
for i, sub in G.items():
    edge_weight = [(i, x, {"weight": v}) for x, v in sub.items()]
    graph.add_edges_from(edge_weight)

pos = nx.spring_layout(graph)
nx.draw_networkx_nodes(graph, pos, node_shape='o', node_size=300)
nx.draw_networkx_edges(graph, pos)
nx.draw_networkx_labels(graph, pos, font_size=10)
nx.draw_networkx_edge_labels(
    graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'))
# nx.draw(graph,with_labels=True)
dis = Dijkstra(G,v0=1)
print dis.values()
plt.show()


# start=1
# end=6
# path={(start,):0} #已选路径
# unsel=G.keys() #未选集合
# unsel.remove(start)
# while True:
#     #如果未选集合为空
#     if len(unsel)==0:
#         break
#     for p,w in path.items():
#         index=p[-1]
#         for k,v in G[index]:
#             k1=p+(,k)
#             v1=w+v
#             if k1 in path.keys() and v1<path[k1]:
#                 path[k1]=v1
#                 del

#     unsel.remove(min_id)

#计算1：求无向图的任意两点间的最短路径
# G = nx.DiGraph()
# G.add_path([5, 6, 7, 8])
# sub_graph = G.subgraph([5, 6, 8])
# #sub_graph = G.subgraph((5, 6, 8))  #ok  一样

# # nx.draw(sub_graph)
# nx.draw(G,with_labels=True)
# plt.show()