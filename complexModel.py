import random as rd
import math
import csv
import matplotlib.pyplot as plt
from decimal import Decimal

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
        #theta = math.acos((v_1[0] * v_2[0] + v_1[1] * v_2[1]) / math.pow(temp, 0.5))
        cos = (v_1[0] * v_2[0] + v_1[1] * v_2[1]) / math.pow(temp, 0.5)
        feedback_corner = cos * 10
        if(flag_temp == "corner"):
            return feedback_corner
        else:
            dis = math.pow(math.pow(Pairs[self.action][0] - Pairs[self.node_curr][0], 2) + math.pow(Pairs[self.action][1] - Pairs[self.node_curr][1], 2), 0.5)
            feedback_up = -dis * 1
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

#数据输入
def input_data(file_path):
    global G
    global Pairs
    global Nodes
    file_file = open(file_path)
    file_csv = csv.reader(file_file, delimiter=' ')
    for line in file_csv:                   
        len_temp = len(line)
        if(len_temp == 1):
            Nodes = int(line[0])
            for i in range(Nodes):
                temp = []
                Pairs.append(temp)
                for j in range(Nodes):
                    temp.append(0)
                G.append(temp)
        elif(len_temp == 3):
            Pairs[int(line[0])] = [float(line[1]) + 30, float(line[2]) + 10]
        else:
            G[int(line[0])][int(line[1])] = 1
            G[int(line[1])][int(line[0])] = 1
    file_file.close()

#Q转弯
def Q_corner(state, graph, EPISODE):
    global Nodes
    global Pool
    Q = {}  
    GREED = 0.3
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
    EPISODE = 100
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
            temp = Q_corner(jum.state + (jum.action,), jum.g, 200)
            if(Q.get(jum.state, 0) == 0):
                Q[jum.state] = {}
            Q[jum.state][jum.action] = jum.feedback("up")+ temp[1] + jum.Qmax(Q, temp[0])
            jum.move_plus(temp[0])
        Q_plt.append((max(Q[state].values()) if Q.get(state, 0) != 0 else 0) + feedback_temp)      #存储奖励的累计值, 三元运算
        time += 1         
    #画图
    #plt.figure()
    #plt.plot(Q_plt)
    #plt.title(Answers_size)
    #plt.show()
    return jum.state, Q_plt[EPISODE - 1]         #返回一次搜寻的结果

#核心搜索函数
def Search(start):
    global G
    global Nodes
    global Pairs
    Answers_state = []
    Answers_q = []
    Answers_size = 50
    while(Answers_size > 0):
        temp = Q_corner((-1,start), G, 2000)
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

#输出路线点
def get_path(state, g, layer):
    global Pairs
    route = [1]
    for i in range(len(state) - 1):
        A = state[i]
        B = state[i+1]
        ad(route, Pairs[A])
        route.append(Pairs[B])
        if(g[A][B] == 0):
            Plane[(layer, len(route)-3, len(route)-2)] = 1
        g[A][B] = 0
        g[B][A] = 0
    return route[1:]

def G_copy():
    global G
    g = []
    for i in G:
        g.append(list(i))
    return g 

def ad(r, a):
    if(r[-1] != a):
        r.append(a)

def get_dis(a, b):
    temp = math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2)
    return math.pow(temp, 0.5)
  
#保留小数位数
def standard():
    global Route
    for i in Route:
        for j in i:
            j[0] = Decimal(j[0]).quantize(Decimal('0.000'))
            j[1] = Decimal(j[1]).quantize(Decimal('0.000'))

#gcode部分
def gcode():
    global Route
    global Plane
    z = 0.2  #层厚
    t1 = 0.0699285714   #内部点的挤出率
    t2 = 0.0758928571   #轮廓点的挤出率
    A = 0
    txt = open('work/mult.gcode', mode='x')
    txt.write('G1 X24' + ' Y10 Z0.5 F300' + '\n')       #打印头从初始位置移动到此位置
    for i in range(18):    #立方体底座
        A += get_dis([i*4 + 24, 20], [i*4 + 24, 80]) * 0.2
        txt.write('G1 X' + str(i*4+24) + ' Y70 Z0.5 F1000 A' + str(Decimal(A).quantize(Decimal('0.00000'))) + '\n')
        A += get_dis([i*4 + 24, 80], [i*4 + 26, 80]) * 0.2
        txt.write('G1 X' + str(i*4+26) + ' Y70 Z0.5 F300 A' + str(Decimal(A).quantize(Decimal('0.00000'))) + '\n')
        A += get_dis([i*4 + 26, 80], [i*4 + 26, 20]) * 0.2
        txt.write('G1 X' + str(i*4+26) + ' Y10 Z0.5 F1000 A' + str(Decimal(A).quantize(Decimal('0.00000'))) + '\n')
        A += get_dis([i*4 + 26, 20], [i*4 + 28, 20]) * 0.2
        txt.write('G1 X' + str(i*4+28) + ' Y10 Z0.5 F300 A' + str(Decimal(A).quantize(Decimal('0.00000'))) + '\n')  
    txt.write('G1 X' + str(Route[0][0][0]) + ' Y' + str(Route[0][0][1]) + ' Z0.5 F9000;travel move' + '\n')   #移动到第一个点
    for i in range(len(Route)):       #z层厚 t挤出率
        Z = Decimal(0.7+i*z).quantize(Decimal('0.000'))
        #txt.write('layer:' + str(i) + '\n')
        txt.write('M73 P' + str(Decimal((i*100) / len(Route)).quantize(Decimal('0.000'))) + '\n')   
        #路线点
        for j in range(1, len(Route[i]), 1):
            if(Plane.get((i, j - 1, j), 0) == 0 ):
                A += get_dis(Route[i][j - 1], Route[i][j]) * t1
                txt.write('G1 X' + str(Route[i][j][0]) + ' Y' + str(Route[i][j][1]) + ' Z' + str(Z) + ' F4200' + ' A' + str(Decimal(A).quantize(Decimal('0.00000'))) + ';infill' + '\n')
            else:
                txt.write('G1 X' + str(Route[i][j][0]) + ' Y' + str(Route[i][j][1]) + ' Z' + str(Z) + ' F9000;travel move' + '\n')
                print(i,' ',"Plane：",j - 1,'  ',j)            #############    
    txt.close()
    
#主函数    
Start = 0
Route = []
Plane = {}
for i in range(1):
    print("当前是第",i,"层")
    G = []
    Pool = {}
    Pairs = []
    Nodes = 0
    input_data("data/data77913/test6.csv")
    g = G_copy()
    state = Search(Start)[1:]
    Route.append(get_path(state, g, i))
    Start = state[-1]
    
#保留小数位数
#生成gcode
standard()
gcode()
