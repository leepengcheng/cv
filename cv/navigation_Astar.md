### 全局路径规划采用A*算法
设起点的的坐标为start,终点坐标为goal  
每个网格点到终点的权重评分
```
fScore=g_score+h_score
```
其中
g_score：当前点到沿着start点A产生的路径到A点的移动耗费   
h_score：不考虑不可通过区域，当前点到goal点B的理论移动耗费   
```bash

**伪代码实现**  

function A*(start, goal)
    //初始化关闭列表，已判定过的节点，进关闭列表。
    closedSet := {}
    // 初始化开始列表，待判定的节点加入开始列表。
    // 初始openset中仅包括start点。
    openSet := {start}
    // 对每一个节点都只有唯一的一个父节点，用cameFrom集合保存节点的子父关系。    
         //cameFrom（节点）得到父节点。
    cameFrom := the empty map

    // gScore估值集合
    gScore := map with default value of Infinity
    gScore[start] := 0 

    // fScore估值集合
    fScore := map with default value of Infinity
    fScore[start] := heuristic_cost_estimate(start, goal)

    while openSet is not empty
                    //取出F值最小的节点设为当前点
        current := the node in openSet having the lowest fScore[] value
                    //当前点为目标点，跳出循环返回路径
        if current = goal
            return reconstruct_path(cameFrom, current)

        openSet.Remove(current)
        closedSet.Add(current)

        //neighbor不包括障碍点
        for each neighbor of current
            // 忽略关闭列表中的节点
            if neighbor in closedSet
                continue        
            // tentative_gScore作为新路径的gScore
            tentative_gScore := gScore[current] + dist_between(current, neighbor)
            if neighbor not in openSet    
                openSet.Add(neighbor)
            else if tentative_gScore >= gScore[neighbor]
                continue        //新gScore>=原gScore，则按照原路径

            // 否则选择gScore较小的新路径，并更新G值与F值。同时更新节点的父子关系。
            cameFrom[neighbor] := current
            gScore[neighbor] := tentative_gScore
            fScore[neighbor] := gScore[neighbor] + heuristic_cost_estimate(neighbor, goal)

    return failure
    //从caomeFrom中从goal点追溯到start点，取得路径节点。
function reconstruct_path(cameFrom, current)
    total_path := [current]
    while current in cameFrom.Keys:
        current := cameFrom[current]
        total_path.append(current)
    return total_path
```