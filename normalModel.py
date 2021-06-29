import random as rd
import math
import csv
import matplotlib.pyplot as plt
from decimal import Decimal

G = []
Pairs = []
Nodes = 0
Pool = {}

#定义打印机类
class Printer:
    def __init__(self, state, graph):
        self.state = state
        self.node_curr = 0
        self.node_last = 0
        self.actions = []
        self.action = 0
        self.g = []
        for i in range(Nodes):
            self.g.append(list(graph[i]))

    def get_node(self):
        self.node_curr = self.state[-1]
        self.node_last = self.state[-2]

    def get_actions_corner(self):
        temp = []
        for i in range(Nodes):
            if(self.g[self.node_curr][i] == 1 and i != self.node_last):
                temp.append(i)
        self.actions = temp
    
    def get_actions_up(self):
        temp = set()
        for i in range(Nodes):
            for j in range(i):
                if(self.g[i][j] == 1):
                    temp.add(i)
                    temp.add(j)
        self.actions = list(temp)
    
    def select_action(self, Q_temp, greed_temp):
        if(Q_temp.get(self.state, 0) == 0):
            self.action = rd.choice(self.actions)
        else:
            actions_greed = []
            dict_temp = Q_temp[self.state]
            max_temp = max(dict_temp.values())
            for i in dict_temp:
                if(dict_temp[i] == max_temp):
                    actions_greed.append(i)
            actions_rd = []
            for i in self.actions:
                if(not (i in actions_greed)):
                    actions_rd.append(i)
            if(actions_rd == []):
                self.action = rd.choice(self.actions)
            else:
                if(rd.random() < greed_temp):
                    self.action = rd.choice(actions_greed)
                else:
                    self.action = rd.choice(actions_rd)
    
    def feedback(self, flag_temp):
        if(self.node_last == -1):
            return 0.0
        v_1 = [Pairs[self.node_curr][0] - Pairs[self.node_last][0], Pairs[self.node_curr][1] - Pairs[self.node_last][1]]
        v_2 = [Pairs[self.action][0] - Pairs[self.node_curr][0], Pairs[self.action][1] - Pairs[self.node_curr][1]]
        temp = (math.pow(v_1[0], 2) + math.pow(v_1[1], 2)) * (math.pow(v_2[0], 2) + math.pow(v_2[1], 2))
        theta = math.acos((v_1[0] * v_2[0] + v_1[1] * v_2[1]) / math.pow(temp, 0.5) / 2)   ################    
        feedback_corner = - theta + 3.15
        if(flag_temp == "corner"):
            return feedback_corner
        else:
            dis = math.pow(math.pow(Pairs[self.action][0] - Pairs[self.node_curr][0], 2) + math.pow(Pairs[self.action][1] - Pairs[self.node_curr][1], 2), 0.5)
            feedback_up = -dis * 0.05
            return feedback_corner * 0.2 + feedback_up

    def move(self):
        self.state += (self.action,)
        self.g[self.node_curr][self.action] = 0
        self.g[self.action][self.node_curr] = 0

    def Qmax(self, Q_temp, state_temp):   
        if(Q_temp.get(state_temp, 0) == 0):
            return 0.0
        else:
            return max(Q_temp[state_temp].values())
    
    def move_plus(self, state_temp):
        self.state = state_temp
        len_temp = len(state_temp)
        for i in range(2,len_temp):
            self.g[state_temp[i]][state_temp[i - 1]] = 0
            self.g[state_temp[i - 1]][state_temp[i]] = 0 

#算法核心函数
#Q转弯
def Q_corner(state, graph):
    global Nodes
    global Pool
    Q = {}  
    GREED = 0.3
    EPISODE = 30
    greed_add = (0.9 - GREED) / EPISODE
    if(Pool.get(state, 0) != 0):
        return Pool[state]
    while(EPISODE > 0):
        pri = Printer(state, graph)
        while(True):
            pri.get_node()
            pri.get_actions_corner()
            if(pri.actions == []):
                GREED += greed_add
                if(EPISODE == 2):
                    GREED = 1.0
                break
            pri.select_action(Q, GREED)
            if(Q.get(pri.state, 0) == 0):
                Q[pri.state] = {}
            Q[pri.state][pri.action] = pri.feedback("corner") + pri.Qmax(Q, pri.state + (pri.action,))
            pri.move()
        EPISODE -= 1
    Pool[state] = [pri.state, max(Q[state].values())]
    return Pool[state]
    
#Q启停
def Q_up(state, graph, feedback_temp, Answers_size):
    global Nodes
    Q = {}  
    GREED = 0.3
    EPISODE = 5
    greed_add = (0.9 - GREED) / EPISODE 
    time = 0
    Q_plt = []
    while(time < EPISODE):
        jum = Printer(state, graph)
        while(True):
            jum.get_node()
            jum.get_actions_up()
            if(jum.actions == []):
                GREED += greed_add
                if(time == EPISODE - 2):
                    GREED = 1.0
                break
            jum.select_action(Q, GREED)
            temp = Q_corner(jum.state + (jum.action,), jum.g)
            if(Q.get(jum.state, 0) == 0):
                Q[jum.state] = {}
            Q[jum.state][jum.action] = jum.feedback("up")+ temp[1] + jum.Qmax(Q, temp[0])
            jum.move_plus(temp[0])
        Q_plt.append((max(Q[state].values()) if Q.get(state, 0) != 0 else 0) + feedback_temp)      #存储奖励的累计值
        time += 1         
    #画图
    #plt.figure()
    #plt.plot(Q_plt)
    #plt.title(Answers_size)
    #plt.show()
    return jum.state, Q_plt[EPISODE - 1]         #返回一次搜寻的结果

#核心搜索函数
def Search(start_node):
    global G
    global Nodes
    global Pairs
    global Pool
    Answers_state = []
    Answers_q = []
    Answers_size = 50
    while(Answers_size > 0):
        temp = Q_corner((-1, start_node), G)
        for i in range(2, len(temp[0])):
            G[temp[0][i]][temp[0][i - 1]] = 0
            G[temp[0][i - 1]][temp[0][i]] = 0 
        answer = Q_up(temp[0], G, temp[1], Answers_size)
        Answers_state.append(answer[0])
        Answers_q.append(answer[1])
        Answers_size -= 1
    plt.figure()
    plt.plot(Answers_q)
    plt.show()
    return Answers_state[Answers_q.index(max(Answers_q))]

  
Edge = []
Outline = []
Connect = []

#数据输入
def input_data():
    global Edge
    global Outline
    txt = open('data/data83378/out_point_5%.txt')
    line = txt.readline().split('\n')[0].split(' ')
    while(True):
        temp = []
        while(True):
            line = txt.readline().split('\n')[0].split(' ')
            if(len(line) == 1 or line == ['']):
                break
            temp.append( [ float(line[1]), float(line[2]), float(line[3]) ])
        Outline.append(temp)
        if(line == ['']):
            break
    txt.close()

    txt = open('data/data83378/graph_point_5%.txt')
    line = txt.readline().split('\n')[0].split(' ')
    while(True):
        G = []
        Pairs = []
        Nodes = 0
        while(True):
            line = txt.readline().split('\n')[0].split(' ')
            len_temp = len(line)
            if(len_temp == 4 or line == ['']):
                break
            if(len_temp == 1):
                Nodes = int(line[0])
                for i in range(Nodes):
                    temp = []
                    Pairs.append(temp)
                    for j in range(Nodes):
                        temp.append(0)
                    G.append(temp)
            elif(len_temp == 3):
                Pairs[int(line[0])] = [float(line[1]), float(line[2])]
            else:
                G[int(line[0])][int(line[1])] = 1
                G[int(line[1])][int(line[0])] = 1
        Edge.append( [Nodes, G, Pairs] )
        if(line == ['']):
            break
    txt.close()

    txt = open('data/data83378/connect_model_5%.txt')
    line = txt.readline().split('\n')[0].split(' ')
    while(line != ['']):
        temp = []
        for i in line:
            temp.append(int(i))
        Connect.append(temp)
        line = txt.readline().split('\n')[0].split(' ')
    txt.close()

#计算IP矩阵
IP = []
def get_IP(edge, outline):
    global IP
    Nodes = edge[0]
    Pairs = edge[2]
    for i in range(Nodes):
        temp = []
        for j in range(Nodes):
            temp.append([-1,-1,-1,-1])
        IP.append(temp)
    #求中心点hub
    a =  99999
    b = -99999
    c =  99999
    d = -99999
    for i in outline:
        a = min(a, i[0])
        b = max(b, i[0])
        c = min(c, i[1])
        d = max(d, i[1])
    hub = [ (a+b)/2, (c+d)/2 ]
    #填数据
    for i in range(Nodes):
        j = (i + 1) % Nodes
        #计算直线距离
        IP[i][j][0] = get_dis(Pairs[i], Pairs[j])
        IP[j][i][0] = IP[i][j][0]
        #计算曲线距离
        nodes = get_inner(i, j, edge, outline)
        IP[i][j][1] = get_curve(nodes)
        IP[j][i][1] = IP[i][j][1]
        #推荐方式
        IP[i][j][2] = get_recommend(i, j, nodes, hub)
        IP[j][i][2] = IP[i][j][2]
        #曲线路径
        IP[i][j][3] = get_route(i, j, nodes, hub)
        temp = list(IP[i][j][3])
        temp.reverse()
        IP[j][i][3] = temp

#util
def get_dis(a, b):
    temp = math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2)
    return math.pow(temp, 0.5)

def get_line(a, b):
    k = ( a[1] - b[1] ) / ( a[0] - b[0] if a[0] != b[0] else 0.00000000000000000000000000001 )
    b = a[1] - k * a[0]
    return [k, b]

def eq(a, b):
    global Error
    if(abs(a - b) <= Error[0]):
        return True
    else:
        return False

def is_in(a, b, x):
    if((x >= a and x <= b) or (x >= b and x <= a)):
        return True
    else:
        return False

def get_inner(A, B, edge, outline):   #A,B是序号，返回[A,B]之间的所有点，方向沿着轮廓点
    global Error
    global Bt
    Nodes_ol = len(outline)
    node_A = []
    node_B = []
    while(node_A == []):
        for i in range(Nodes_ol):
            j = (i + 1) % Nodes_ol
            lij = get_line(outline[i], outline[j])
            if(is_in(outline[i][0], outline[j][0], edge[2][A][0]) and eq(edge[2][A][0] * lij[0] + lij[1], edge[2][A][1])):
                node_A.append(i)
                node_A.append(j)
                break
        Error[0] += Error[1]
    Bt[A] = node_A
    Error = [0.01, 0.005]

    while(node_B == []):
        for i in range(Nodes_ol):
            j = (i + 1) % Nodes_ol
            lij = get_line(outline[i], outline[j])
            if(is_in(outline[i][0], outline[j][0], edge[2][B][0]) and eq(edge[2][B][0] * lij[0] + lij[1], edge[2][B][1])):
                node_B.append(i)
                node_B.append(j)
                break
        Error[0] += Error[1]
    Bt[B] = node_B
    Error = [0.01, 0.005]

    if(node_A == node_B):                        #判断是否在同一条直线上
        return [edge[2][A], edge[2][B]]
    nodes = []
    nodes.append(edge[2][A])
    if(node_B[0] >= node_A[1]):
        for i in range(node_A[1], node_B[0] + 1, 1):
            nodes.append(outline[i])
    else:
        for i in range(node_A[1], node_B[0] + 1 + Nodes_ol, 1):
            nodes.append(outline[i % Nodes_ol])
    nodes.append(edge[2][B]) 

    return nodes

def get_curve(nodes):    #nodes是包含端点的曲线点集合
    dis_sum = 0
    for i in range(len(nodes) - 1):
        dis_sum += get_dis(nodes[i], nodes[i+1])
    return dis_sum

def get_recommend(A, B, nodes, hub):   #A,B是序号
    A = nodes[0]
    B = nodes[-1]
    nodes_cor = []
    nodes = nodes[1:-1]
    if(nodes == []):                              #在一条直线上
        return 0
    lAB = get_line(A, B)
    for i in nodes:
        lihub = get_line(i, hub)
        x = ( lAB[1] - lihub[1] ) / ( lihub[0] - lAB[0] )
        nodes_cor.append([x, x * lAB[0] + lAB[1]])
    flag = 0
    for i in range(len(nodes)):
        if(get_dis(nodes[i], hub) < get_dis(nodes_cor[i], hub)):
            flag = 1
            break
    return flag
        
def get_route(A, B, nodes, hub):     #A,B是序号 
    route = []
    step = 0.3        #控制贴轮廓边距    
    for i in nodes:
        temp = [
            [i[0] + step, i[1]],
            [i[0], i[1] + step],
            [i[0] - step, i[1]],
            [i[0], i[1] - step],
            [i[0] + step, i[1] + step],
            [i[0] - step, i[1] - step],
            [i[0] + step, i[1] - step],
            [i[0] - step, i[1] + step]
        ]
        label = 0
        dis_min = 99999
        for j in range(len(temp)):
            dis = get_dis(temp[j], hub)
            if(dis < dis_min):
                dis_min = dis
                label = j
        route.append(temp[label]) 
    return route 
  
  
#连接
#返回距离最近的奇度点，作为In点
def get_near(S, edge):
    In = 0
    dis_min = 9999999
    for i in range(edge[0]):
        dis = get_dis(S, edge[2][i])
        if(is_odd(i, edge[1]) and dis < dis_min):
            dis_min = dis
            In = i
    return In

#适用于无内部填充线时，外轮廓的排序    
def get_near_outline(S, outline):
    In = 0
    dis_min = 99999999
    for i in range(len(outline)):
        dis = get_dis(S, outline[i])
        if(dis < dis_min):
            dis_min = dis
            In = i
    return In

def get_connect(In, G_temp):       #输入起点，返回最佳填充图
    global IP
    len_IP = len(IP)

    #下一个点为出点，Out大于In
    len_1 = 0
    G_1 = []
    for i in G_temp:
        G_1.append(list(i))
    for i in range(In - 1, In + 1 - len_IP, -1):
        a = ( i + len_IP ) % len_IP
        b = ( a - 1 + len_IP ) % len_IP
        if(is_connect(a, b, G_1)):
            len_1 += ex_connect(a, b, G_1)
    
    #上一个点为出点，Out小于In
    len_2 = 0
    G_2 = []
    for i in G_temp:
        G_2.append(list(i))
    for i in range(In + 1, In + len_IP - 1, 1):
        a = i % len_IP
        b = ( a + 1) % len_IP
        if(is_connect(a, b, G_2)):
            len_2 += ex_connect(a, b, G_2)
            
    print(len_1)
    print(len_2)
    
    if(len_1 < len_2):
        return G_1
    else:
        return G_2

#util
def is_odd(a, G):
    G = G[a]
    count = 0
    for i in G:
        if(i == 1):
            count += 1
    if(count % 2 == 0):
        return False
    else:
        return True

#沿着顺序进行连接，的判断
def is_connect(a, b, G):
    if(is_odd(a, G)):
        return True
    else:
        return False

#连接a，b返回a，b之间的“距离”
def ex_connect(a, b, G):
    global IP
    G[a][b] = 1
    G[b][a] = 1
    return IP[a][b][ IP[a][b][2] ]
  
  
#输出内部路径点
def get_path(state, edge, layer):
    global IP
    global Plane
    G = edge[1]
    Pairs = edge[2]
    route = [1]
    for i in range(len(G)):
        for j in range(len(G)):
            if(G[i][j] == 1):
                IP[i][j][2] = 0
    for i in range(len(state) - 1):
        A = state[i]
        B = state[i+1]
        if(IP[A][B][2] == 0):
            ad(route, [Pairs[A][0], Pairs[A][1]])
            route.append([Pairs[B][0], Pairs[B][1]])
        elif(IP[A][B][2] == 1):
            ad(route, [Pairs[A][0], Pairs[A][1]])
            for j in IP[A][B][3]:
                route.append([j[0], j[1]])
            route.append([Pairs[B][0], Pairs[B][1]])
        else:  #空行程
            ad(route, [Pairs[A][0], Pairs[A][1]])
            route.append([Pairs[B][0], Pairs[B][1]])
            Plane[(layer, len(route)-3, len(route)-2)] = 1
    return route[1:]

#util
def ad(A, a):
    if(A[-1] != a):
        A.append(a)

#输出轮廓点
def get_outline(s, outline):
    global Bt
    nodes = [s[0]]
    i = Bt[s[1]][0]
    j = Bt[s[1]][1]
    if(j > i):
        for k in range(j, i + len(outline) + 1, 1):
            nodes.append(outline[k % len(outline)])
    else:
        for k in range(j, i + 1, 1):
            nodes.append(outline[k])
    return nodes

#适用于以轮廓点为最近点，轮廓的排序
def get_outline_outline(s, outline):
    nodes = []
    for i in range(s, len(outline)):
        nodes.append(outline[i])
    for i in range(s):
        nodes.append(outline[i])
    return nodes

  
#保留三位小数
def standard():
    global Route
    global Outline_order
    for i in Route:
        for j in i:
            j[0] = Decimal(j[0]).quantize(Decimal('0.000'))
            j[1] = Decimal(j[1]).quantize(Decimal('0.000'))
    for i in Outline_order:
        for j in i:
            j[0] = Decimal(j[0]).quantize(Decimal('0.000'))
            j[1] = Decimal(j[1]).quantize(Decimal('0.000'))   

def get_ZS(z):
    global Connect
    ZS = []
    for i in range(len(Connect)):
        for j in Connect[i]:
            ZS.append(0.5 + (i+1)*z)
    return ZS

#自己写的gcode
def gcode(Route, Outline):
    global Plane
    z = 0.2  #层厚
    t1 = 0.0699285714   #内部点的挤出率
    t2 = 0.0699285714
    #t2 = 0.0758928571   #轮廓点的挤出率
    A = 0
    position = 24
    top = 80
    bottom = 20
    txt = open('work/test.gcode', mode='x')
    txt.write('G1 X' + str(position) + ' Y' + str(bottom) + ' Z0.5 F300' + '\n')       #打印头从初始位置移动到此位置
    for i in range(18):    #立方体底座
        A += get_dis([i*4 + position, bottom], [i*4 + position, top]) * 0.2
        txt.write('G1 X' + str(i*4+position) + ' Y' + str(top) + ' Z0.5 F1000 A' + str(Decimal(A).quantize(Decimal('0.00000'))) + '\n')
        A += get_dis([i*4 + position, top], [i*4 + position+2, top]) * 0.2
        txt.write('G1 X' + str(i*4 + position+2) + ' Y' + str(top) + ' Z0.5 F300 A' + str(Decimal(A).quantize(Decimal('0.00000'))) + '\n')
        A += get_dis([i*4 + position+2, top], [i*4 + position+2, bottom]) * 0.2
        txt.write('G1 X' + str(i*4+ position+2) + ' Y' + str(bottom) + ' Z0.5 F1000 A' + str(Decimal(A).quantize(Decimal('0.00000'))) + '\n')
        A += get_dis([i*4 + position+2, bottom], [i*4 + position+4, bottom]) * 0.2
        txt.write('G1 X' + str(i*4+ position+4) + ' Y' + str(bottom) + ' Z0.5 F300 A' + str(Decimal(A).quantize(Decimal('0.00000'))) + '\n')  
    ZS = get_ZS(z)   
    for i in range(len(Outline)):       #z层厚 t挤出率
        Z = Decimal(ZS[i]).quantize(Decimal('0.000'))
        txt.write(';layer:' + str(i) + '\n')
        txt.write('M73 P' + str(Decimal(i / len(Outline)).quantize(Decimal('0.000'))) + '\n')
        if(Route[i] != []):
            #走内部点
            txt.write('G1 X' + str(Route[i][0][0]) + ' Y' + str(Route[i][0][1]) + ' Z' + str(Z) + ' F9000;travel move' + '\n')   #层与层之间的空行程
            for j in range(1, len(Route[i]), 1):
                if(Plane.get((i, j - 1, j), 0) == 0 ):
                    A += get_dis(Route[i][j - 1], Route[i][j]) * t1
                    txt.write('G1 X' + str(Route[i][j][0]) + ' Y' + str(Route[i][j][1]) + ' Z' + str(Z) + ' F2100' + ' A' + str(Decimal(A).quantize(Decimal('0.00000'))) + ';infill' + '\n')
                else:
                    txt.write('G1 X' + str(Route[i][j][0]) + ' Y' + str(Route[i][j][1]) + ' Z' + str(Z) + ' F9000;travel move' + '\n')  #内部图的的空行程
                    print(i,' ',"Plane：",j - 1,'  ',j)            #############
        else:
            txt.write('G1 X' + str(Outline[i][0][0]) + ' Y' + str(Outline[i][0][1]) + ' Z' + str(Z) + ' F9000;travel move' + '\n')   #层与层之间的空行程
        #走外轮廓点
        for j in range(1, len(Outline[i]), 1):
            A += get_dis(Outline[i][j - 1], Outline[i][j]) * t2
            txt.write('G1 X' + str(Outline[i][j][0]) + ' Y' + str(Outline[i][j][1]) + ' Z' + str(Z) + ' F600' + ' A' + str(Decimal(A).quantize(Decimal('0.00000'))) + ';outline' + '\n')
        A += get_dis(Outline[i][len(Outline[i]) - 1], Outline[i][0]) * t2
        txt.write('G1 X' + str(Outline[i][0][0]) + ' Y' + str(Outline[i][0][1]) + ' Z' + str(Z) + ' F600' + ' A' + str(Decimal(A).quantize(Decimal('0.00000'))) + ';outline' + '\n')       
    txt.close()
    
    
    
  
Error = [0.01, 0.005]
Route = []
Plane = {}
Outline_order = []
input_data()
Start = [[96, 20], 0]  #点坐标，点序号
for i in range(482):
    print("当前是第",i,"层")
    if(Edge[i][0] != 0):
        Bt = {}
        IP = []
        get_IP(Edge[i], Outline[i])
        begin = get_near(Start[0], Edge[i])
        G = get_connect(begin, Edge[i][1])
        Nodes = Edge[i][0]
        Pairs = Edge[i][2]
        Pool = {}
        temp = Search(begin)[1:]
        Route.append(get_path(temp, Edge[i], i))
        Start = [Edge[i][2][temp[-1]], temp[-1]]
        Outline_order.append(get_outline(Start, Outline[i]))
    else:
        Route.append([])
        begin = get_near_outline(Start[0], Outline[i])
        Start = [Outline[i][begin], begin]
        Outline_order.append(get_outline_outline(begin, Outline[i]))
print("-----最优路径搜索成功！")


#保留小数位数
#输出gcode
standard()
gcode(Route, Outline_order)
