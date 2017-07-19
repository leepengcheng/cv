#coding:utf-8
#迪杰斯特拉算法:求无向图的最短路径，权值为正
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    # 源顶点到其余各顶点的初始路程
    pre=dict((k, 0) for k in G.keys()) #前驱节点下标
    dis = dict((k, INF) for k in G.keys()) #v0到k的距离
    dis[v0] = 0 #v0到v0距离为0
    sel = [v0] #已求出最短路径的集合

    minv = v0 #当前源点
    while len(sel) < len(G):
        for w in G[minv]:  # 以当前点的中心向外扩散
            if dis[minv] + G[minv][w] < dis[w]:  # 如果从当前点扩展到某一点的距离小与已知最短距离
                dis[w] = dis[minv] + G[minv][w]  # 对已知距离进行更新
                pre[w]=minv #更新前驱节点

        new = INF  # 从剩下的未确定点中选择最小距离点作为新的扩散点
        for v in dis.keys():
            if v in sel: continue 
            if dis[v] < new:
                new = dis[v]
                minv = v
        sel.append(minv)   # 添加该节点到已选集合
    return dis,pre

def Dijkstra_search(pre_vals,start,goal):
    path=[goal]
    pre=goal
    while True:
        #起点为1
        pre=pre_vals[pre-1]
        if pre==0:
            break
        path.append(pre)
    return path[::-1]


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


start,goal=1,4    #起点/终点
dis,pre = Dijkstra(G, v0=start) #距离/前驱字典
dis_vals=dis.values() #距离列表
print dis_vals



pre_vals=pre.values() #前驱列表
path=Dijkstra_search(pre_vals,start,goal)
print path



#绘制图
graph = nx.Graph()
for i, sub in G.items():
    edge_weight = [(i, x, {"weight": v}) for x, v in sub.items()]
    graph.add_edges_from(edge_weight)
pos = nx.spring_layout(graph)
nx.draw_networkx_nodes(graph, pos, node_shape='o', node_size=300)
nx.draw_networkx_edges(graph, pos)
nx.draw_networkx_labels(graph, pos, font_size=10)
nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'weight'))
# nx.draw(graph,with_labels=True)
plt.show()
