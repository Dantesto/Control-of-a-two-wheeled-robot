import math
import sys
import tkinter
import time
import collections
import heapq
import matplotlib.pyplot as plt
import numpy as np
import random

dt = 0.001 #delta t 0.001
x_r = 0.01 #diameter of x's neighbourhood (диаметр окрестности икса состояния робота) в метрах
dx_r = 0.1
theta_r = 2 * math.pi / 360 #один градус
dtheta_r = 0.1
tau_set = [x / 10 for x in range(-20, 20)] #набор моментов, которые можно подать
#tau_set = [-1, 0, 1]
g_max_size = 1000 #максимальное количество вершин графа g
g_max_memory = 1073741824 #максимальный объем занимаемой памяти графом g, 1 гигабайт

q_learning_episodes = 10000
q_learning_episode_max_steps = 1000
q_learning_alpha = 0.1 #Коэффициент обучения (learning rate)
q_learning_gamma = 0.1 #Коэффициент важности будущих наград (discount factor)
exploring_rate = 0.1 #Коэффициент исследования, вероятность выбора не жадной стратегии (epsilon-greedy)

m = 1.2 #kg - chassis weight
M = 0.02 #kg - wheel weight
J_c = 0.015 #kgm^2 - chassis inertia
J_w = 0.00002 #kgm^2 - wheel inertia
l = 0.075 #m - chassis length
R = 0.032 #m - wheel radius
mu_0 = 0.1 #coefficient of friction between wheel and ground
mu_1 = 0 #coefficient of friction between chassis and wheel
g = 9.81 #acceleration of gravity
brake_rate = 0.5 #вероятность поломки, в результате которой будет выполнено движение с меньшим моментом, чем который был подан
tau_change_rate = 0.5 #множитель при старом моменте во время поломки

class RobotState:
    def __init__(self, x, dx, theta, dtheta):
        self.x = x
        self.dx = dx
        self.theta = theta
        self.dtheta = dtheta
    def __str__(self):
        return str((self.x, self.dx, self.theta, self.dtheta))
    def __hash__(self):
        c_1 = 1664525
        c_2 = 1013904223
        cn_x = CellNum(self.x, x_r)
        cn_dx = CellNum(self.dx, dx_r)
        cn_theta = CellNum(self.theta, theta_r)
        cn_dtheta = CellNum(self.dtheta, dtheta_r)
        #return hash((((cn_x + c_2) * c_1 + cn_dx + c_2) * c_1 + cn_theta + c_2) * c_1 + cn_dtheta + c_2)
        return hash((cn_x, cn_dx, cn_theta, cn_dtheta))
    #неправльно хэш считаю. Нужно брать не значения координат, а номера клеток, в которых они. Через CellNum из IsEqualStates

rs_default = RobotState(0, 0, 0, 0)

def WayNorm(m, t): #длина пути (норма)
    return abs(m) #не учитывая, время, это просто величина момента

def GetMinIndVertex(q): #возвращает индекс вершины с минимальным до нее расстоянием от начальной
    min_ind = 0
    for i in range(len(q)):
        if (WayNorm(q[i][1], q[i][2]) < WayNorm(q[min_ind][1], q[min_ind][2])):
            min_ind = i
    return min_ind

def GetNextVertex(rs_cur, tau):
    ddtheta = (m * l * math.cos(rs_cur.theta) * (tau / R - 2 * mu_0 * rs_cur.dx + m * l * math.sin(rs_cur.theta) * rs_cur.dtheta**2) - (m + 2 * M + 2 * J_w / R**2) * (-tau - 2 * mu_1 * rs_cur.dtheta + m * g * l * math.sin(rs_cur.theta))) / ((m * l * math.cos(rs_cur.theta))**2 - (m + 2 * M + 2 * J_w / R**2) * (m * l**2 + J_c)) #Q2 = -tau
    #ddtheta = (m * l * math.cos(rs_cur.theta) * (tau / R - 2 * mu_0 * rs_cur.dx + m * l * math.sin(rs_cur.theta) * rs_cur.dtheta**2) - (m + 2 * M + 2 * J_w / R**2) * (0 - 2 * mu_1 * rs_cur.dtheta + m * g * l * math.sin(rs_cur.theta))) / ((m * l * math.cos(rs_cur.theta))**2 - (m + 2 * M + 2 * J_w / R**2) * (m * l**2 + J_c)) #Q2 = 0
    ddx = (-tau - (m * l**2 + J_c) * ddtheta - 2 * mu_1 * rs_cur.dtheta + m * g * l * math.sin(rs_cur.theta)) / (m * l * math.cos(rs_cur.theta)) #Q2 = -tau
    #ddx = (0 - (m * l**2 + J_c) * ddtheta - 2 * mu_1 * rs_cur.dtheta + m * g * l * math.sin(rs_cur.theta)) / (m * l * math.cos(rs_cur.theta)) #Q2 = 0
    x = rs_cur.x + dt * rs_cur.dx
    dx = rs_cur.dx + dt * ddx
    theta = rs_cur.theta + dt * rs_cur.dtheta
    dtheta = rs_cur.dtheta + dt * ddtheta
    rs_next = RobotState(x, dx, theta, dtheta)
    return rs_next

CellNum = lambda param, step: math.trunc(param / step) if param >= 0 else math.trunc(param / step) - 1 #для случая 0.5 и -0.5 trunc(0.5)=trunc(-0.5). Поэтому вычитаю 1 из отрицательных

def IsEqualStates(rs_1, rs_2, important=[True, True, True, True]):
    return (((CellNum(rs_1.x, x_r) == CellNum(rs_2.x, x_r)) or (not important[0]))
         and ((CellNum(rs_1.dx, dx_r) == CellNum(rs_2.dx, dx_r)) or (not important[1]))
         and ((CellNum(rs_1.theta, theta_r) == CellNum(rs_2.theta, theta_r)) or (not important[2]))
         and ((CellNum(rs_1.dtheta, dtheta_r) == CellNum(rs_2.dtheta, dtheta_r)) or (not important[3])))

def GetVertexInd(g, hs, rs):
    if (len(hs) > 0) and not (hash(rs) in hs):
        return len(g)
    for i in range(len(g)):
        if (IsEqualStates(g[i][0], rs)):
            return i
    return len(g)

def ContainEdge(e, v_ind):
    for (i, w) in e:
        if (i == v_ind):
            return True
    return False

def CopyGraph(g):
    g_copy = []
    for (v, es) in g:
        g_copy.append((RobotState(v.x, x.dx, v.theta, v.dtheta), []))
        for (ind, w) in es:
            g_copy[-1][1].append((ind, w))
    return g_copy

def ReverseGraph(g):
    g_reversed = []
    for (v, es) in g:
        g_reversed.append((RobotState(v.x, v.dx, v.theta, v.dtheta), []))
    for i in range(len(g)):
        for (ind, w) in g[i][1]:
            g_reversed[ind][1].append((i, w))
    return g_reversed

#Возвращает граф с неотрицательными ребрами
def BuildGraph(rs_0=rs_default):
    g = [(rs_0, [])] #vertex, list of edges (ind in g of neighbour, weight)
    q = collections.deque() #номер непросмотренной вершины в g, то есть очередь
    q.append(0)
    hs = {hash(rs_0)} #hashset of rs in g
    while (len(q) != 0):
        i_cur = q.popleft() #index of current vertex in graph g
        rs_cur = g[i_cur][0]
        rs_prev = rs_cur
        for tau in tau_set:
            rs_next = GetNextVertex(rs_cur, tau)
            if (IsEqualStates(rs_cur, rs_next)) or (IsEqualStates(rs_prev, rs_next)) or (abs(rs_next.theta) >= math.pi / 2): #не создаем петли, кратные ребра и углы больше 90 градусов
                continue
            rs_prev = rs_next
            i_next = GetVertexInd(g, hs, rs_next) #index of rs_next in g
            if (i_next < len(g)):
                #if (not ContainEdge(g[i_cur][1], i_next)): вроде как проверка на существование ребра не нужна, так как я проверяю совпадает ли предыдущая новая вершина rs_prev с текущей новой rs_next
                #    g[i_cur][1].append((i_next, WayNorm(tau, dt)))
                g[i_cur][1].append((i_next, WayNorm(tau, dt)))
            elif (len(g) < g_max_size): #или sys.getsizeof(g) < g_max_memory, если эта функция быстро работает
                g.append((rs_next, []))
                g[i_cur][1].append((i_next, WayNorm(tau, dt)))
                q.append(i_next)
                hs.add(hash(rs_next))
    return g

#Получаем новое действие в результате поломки. v_i - индекс текущей вершины, e_i - индекс текущего ребра
#Возвращает True, если момент изменился, и значение момента
def GetChangedAction(g, v_i, e_i):
    if (random.random() < brake_rate):
        changed_tau = tau_change_rate * g[v_i][1][e_i][1]
        e_c = 0
        for i in range(len(g[v_i][1])): #ищем ближайшее по величине момента ребро к changed_tau
            if (abs(g[v_i][1][i][1] - changed_tau) < abs(g[v_i][1][e_c][1] - changed_tau)) and (i != e_i):
                e_c = i
        return True, e_c
    return False, e_i

def BFS(g, i_i, i_g):
    q = collections.deque()
    q.append(i_i)
    visited = [False] * len(g)
    visited[i_i] = True
    dist = [0] * len(g)
    prev_vertex = [-1] * len(g)
    while (len(q) != 0):
        i_cur = q.popleft()
        v_cur = g[i_cur]
        for (i_next, _) in v_cur[1]:
            if (not visited[i_next]):
                q.append(i_next)
                visited[i_next] = True
                dist[i_next] = dist[i_cur] + 1
                prev_vertex[i_next] = i_cur
    return prev_vertex, dist

def DijkstraAlgorithm(g, i_i, i_g):
    #а еще стоит принимать третьим параметром rs_g, чтобы как только его вытащили из кучи, вернуть результат
    h = [(0, i_i)] #расстояние (NormWay - в данный момент просто суммарный момент), номер вершины. надо свою кучу написать, которая будет хранить только номер вершины, чтобы проверка на расстояние шла по массиву dist
    heapq.heapify(h)
    visited = [False] * len(g)
    dist = [math.inf] * len(g)
    dist[i_i] = 0
    prev_vertex = [-1] * len(g)
    while (len(h) != 0):
        (d_cur, i_cur) = heapq.heappop(h)
        if (visited[i_cur]):
            continue
        visited[i_cur] = True
        v_cur = g[i_cur]
        for (i_next, weight_to_next) in v_cur[1]:
            d_new = dist[i_cur] + weight_to_next
            if (d_new < dist[i_next]):
                dist[i_next] = d_new
                heapq.heappush(h, (d_new, i_next))
                prev_vertex[i_next] = i_cur
    return prev_vertex, dist

#Инициализация функции Q из графа g
def InitializeQ(g):
    Q = []
    for (v, es) in g:
        Q.append([RobotState(v.x, v.dx, v.theta, v.dtheta), []])
        for (ind, w) in es:
            #Q[-1][1].append([ind, random.random()]) #Заполнение весов Q[s,a] случайными числами
            Q[-1][1].append([ind, 0]) #Заполнение весов Q[s,a] нулями
    return Q

#s - индекс состояния (вершины)
#Возвращает индекс действия (ребра)
def GetAction(Q, s, eps=0.1):
    if (random.random() > eps):
        max_action = 0
        for i in range(len(Q[s][1])):
            if (Q[s][1][i][1] > Q[s][1][max_action][1]):
                max_action = i
        return max_action
    return random.randint(0, len(Q[s][1]) - 1)

def QLearning(g, i_i, i_g): #По сути функция Q - тот же граф, только веса она ставит сама на ребра
    global q_learning_gamma
    global exploring_rate
    q_learning_gamma = 0 #Начальное значение q_learning_gamma. Оно сразу будет увеличено в цикле
    exploring_rate = 1.1
    Q = InitializeQ(g) #Q function
    reward_func = DijkstraAlgorithm(ReverseGraph(g), i_g, i_i)[1] #Разворачиваем граф и запускаем Дейкстру из конечного состояния. Теперь наградой будет отрицательное значение длины пути от текущей точки до конечной.
    reward_norm = 1 #Максимальное расстояние среди всех, кроме бесконечности
    for i in range(len(reward_func)):
        if (reward_func[i] > reward_norm) and (math.isfinite(reward_func[i])):
            reward_norm = reward_func[i]
    for i in range(q_learning_episodes):
        if (i % (q_learning_episodes // 100) == 0): #Увеличиваем со временем q_learning_gamma. То есть со временем начинаем больше верить старой информации.
            q_learning_gamma += 0.01
        if (i < q_learning_episodes // 10) and (i % (q_learning_episodes // 100) == 0): #Уменьшаем со временем exploring_ratee. То есть по-началу чаще выбираем случайное действие, а не жадное.
            exploring_rate -= 0.1
        s_prev = random.randint(0, len(Q) - 1)
        #s_prev = i_i #Здесь идея обучать только из нужного начального положения
        if (len(Q[s_prev][1]) == 0):
            continue
        a_prev = GetAction(Q, s_prev, exploring_rate)
        steps = 0
        while (steps < q_learning_episode_max_steps):
            s_cur = Q[s_prev][1][a_prev][0]
            if (len(Q[s_cur][1]) == 0):
                break
            a_cur = GetAction(Q, s_cur, -1)
            r = -reward_func[s_cur] / reward_norm #Наградой будет отрицательное расстояние от текущей вершины до конечной
            Q[s_prev][1][a_prev][1] += q_learning_alpha * (r + q_learning_gamma * Q[s_cur][1][a_cur][1] - Q[s_prev][1][a_prev][1])
            a_cur = GetAction(Q, s_cur, exploring_rate)
            a_cur = GetChangedAction(g, s_cur, a_cur)[1] #Меняем момент в случае проскальзывания
            s_prev = s_cur
            a_prev = a_cur
            steps += 1
    return Q

        #Каждая вершина графа - координата, скорость, угол, угловая скорость.
        #В текущий момент (в текущей вершине) ускорение не определено, потому что
        #следующие состояния робота (скорость, координата и прочее) определяются
        #ускорением на предыдущем шаге, которое зависит от момента. Если бы каждая
        #вершина также хранила фиксированое ускорение, то подача момента влияла бы
        #только на ускорение в будущем состоянии, а на другие параметры не влияла бы.
        #Кроме того, тогда в одну и ту же вершину нельзя было бы попасть из разных мест.
        #Например в таком случае, в вершину i (x_i, dx_i, ddx_i, t_i, dt_i, ddt_i) можно попасть, задав
        #конкретный момент, который выражается из формул ddx и ddt из моего файла
        #"Вывод ДУ движения", так как (x_i, dx_i, t_i, dt_i) однозначно определяются
        #ddx_i-1 и ddt_i-1, а фиксированые (именно фиксированые) ddx_i и ddt_i однозначно
        #опрделяются (x_i, dx_i, t_i, dt_i) и моментом. А раз ddx_i и ddt_i фиксированы,
        #то и момент может быть только один, то есть другие ребра не могут вести в эту вершину.

        #И еще. Скорее всего Q2 = -tau, как в файле Modeling_of_Two-Wheeled_Self-Balancing_Robot_Drive.
        #Логика подсказывает, что все же момент должен отклонять палку назад. Но можно будет проверить
        #поведение и с нулевым моментом.

        #Суммировать моменты нужно по модулю.

        #выровнять робота, уменьшить скорость до нуля
        #выравнивание, когда выравнивается и останавливается. выравнивание, когда выравнивается и дальше едет

#Находит точку, соответсвующую галочкам в чекбоксах
def GetBoundaryPointInd(g, rs, important):
    for i in range(len(g)):
        if (IsEqualStates(g[i][0], rs, important)):
            return i
    return -1

def RecoverWay(g, prev_vertex, i_i, i_g):
    way = []
    while (i_g != i_i):
        way.append(g[i_g][0])
        i_g = prev_vertex[i_g]
    way.append(g[i_g][0])
    way.reverse()
    return way

def RecoverEdgeWay(g, prev_vertex, i_i, i_g):
    edges_ind = [] #Индексы ребер на пути
    i_next = i_g
    i_cur = prev_vertex[i_g]
    while (i_cur != -1):
        for i in range(len(g[i_cur][1])):
            if (g[i_cur][1][i][0] == i_next):
                edges_ind.append(i)
                break
        i_next = i_cur
        i_cur = prev_vertex[i_cur]
    edges_ind.reverse()
    return edges_ind

def RecoverWayQ(g, Q, i_i, i_g):
    way = []
    max_steps = len(g)
    steps = 0
    while (i_i != i_g) and (steps < max_steps):
        way.append(graph[i_i][0])
        a = GetAction(Q, i_i, -1)
        if (len(Q[i_i][1]) == 0):
            break
        i_i = Q[i_i][1][a][0]
        steps += 1
    if (i_i == i_g):
        way.append(graph[i_i][0])
    return way

class GraphicalModel:
    def __init__(self):
        self.scale = 2000
        self.root = tkinter.Tk()
        self.root.title("Motion modeling")
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.attributes("-fullscreen", True)
        self.root.geometry("%dx%d+%d+%d" % (sw, sh - 70, -10, 0))
        self.input_frame = tkinter.Frame(self.root)
        self.input_frame.pack(anchor=tkinter.NE)
        self.xi_label = tkinter.Label(self.input_frame, text="x_i = ")
        self.xi_label.grid(column=0, row=0)
        self.xg_label = tkinter.Label(self.input_frame, text="x_g = ")
        self.xg_label.grid(column=0, row=1)
        self.xi_entry = tkinter.Entry(self.input_frame, width=10)
        self.xi_entry.grid(column=1, row=0)
        self.xg_entry = tkinter.Entry(self.input_frame, width=10)
        self.xg_entry.grid(column=1, row=1)
        self.important_label = tkinter.Label(self.input_frame, text="Mark important params")
        self.important_label.grid(column=2, row=0, columnspan=4)
        self.important_checkbutton_state = [tkinter.BooleanVar() for _ in range(4)]
        self.important_checkbutton = [tkinter.Checkbutton(self.input_frame, var=self.important_checkbutton_state[i]) for i in range(4)]
        for i in range(4):
            self.important_checkbutton[i].grid(column=i+2, row=1)
        self.start_button = tkinter.Button(self.input_frame, text="Start", command=self.StartModeling)
        self.start_button.grid(column=0, row=2)
        self.quit_button = tkinter.Button(self.input_frame, text="Quit", command=self.root.quit)
        self.quit_button.grid(column=1, row=2)
        self.model_canvas = tkinter.Canvas(self.root, background="white")
        self.model_canvas.pack(expand=True, fill=tkinter.BOTH)
        self.model_canvas.create_line(0, sh - 175, sw, sh - 175, dash=(1, 1))
        self.wheel_oval = self.model_canvas.create_oval(sw / 2 - R * self.scale, sh - 175 - 2 * R * self.scale, sw / 2 + R * self.scale, sh - 175, outline="blue")
        self.wheel_angle_line = self.model_canvas.create_line(sw / 2, sh - 175 - R * self.scale, sw / 2, sh - 175 - 2 * R * self.scale, fill="blue")
        self.chassis_rect = self.model_canvas.create_line(sw / 2 - 0.5 * R * self.scale, sh - 175 - R * self.scale, sw / 2 - 0.5 * R * self.scale, sh - 175 - (R + l) * self.scale, sw / 2 + 0.5 * R * self.scale, sh - 175 - (R + l) * self.scale, sw / 2 + 0.5 * R * self.scale, sh - 175 - R * self.scale, sw / 2 - 0.5 * R * self.scale, sh - 175 - R * self.scale, fill="red")
        #self.state_text = self.model_canvas.create_text(500, 500)
        self.root.mainloop()
    def StartModeling(self): #chosen_method - функция построения пути робота
        #сделать кнопки quit и start неактивными
        xi = [float(param) for param in self.xi_entry.get().split()]
        xg = [float(param) for param in self.xg_entry.get().split()]
        rs_i = RobotState(xi[0], xi[1], xi[2], xi[3])
        rs_g = RobotState(xg[0], xg[1], xg[2], xg[3])
        important = [self.important_checkbutton_state[i].get() for i in range(len(self.important_checkbutton_state))]
        i_i = GetBoundaryPointInd(graph, rs_i, important)
        i_g = GetBoundaryPointInd(graph, rs_g, important)
        if (i_i == -1) or (i_g == -1):
            print("not found states")
            return
        if (BFS(graph, i_i, i_g)[0][i_g] == -1):
            print("not found path")
            return
        print(graph[i_i][0])
        print(graph[i_g][0])
        if (chosen_method.__name__ == "BFS") or (chosen_method.__name__ == "DijkstraAlgorithm"):
            prev_vertex = chosen_method(graph, i_i, i_g)[0]
            way = RecoverWay(graph, prev_vertex, i_i, i_g)
        if (chosen_method.__name__ == "QLearning"):
            way = RecoverWayQ(graph, QLearning(graph, i_i, i_g), i_i, i_g)
        #поставить робота в начальное состояние way[0]
        print(len(way))
        for i in range(1, len(way)):
            print(way[i])
            self.model_canvas.move(self.wheel_oval, (way[i].x - way[i - 1].x) * self.scale, 0)
            wheel_oval_coords = self.model_canvas.coords(self.wheel_oval)
            wheel_angle = (way[i].x - way[0].x) / R #x = phi * R для движения без проскальзывания
            self.model_canvas.coords(self.wheel_angle_line, wheel_oval_coords[0] + R * self.scale, wheel_oval_coords[1] + R * self.scale, wheel_oval_coords[0] + R * self.scale * (math.sin(wheel_angle) + 1), wheel_oval_coords[1] + R * self.scale * (1 - math.cos(wheel_angle)))
            chassis_angle = way[i].theta
            ldx = wheel_oval_coords[0] + R * self.scale * (1 - 0.5 * math.cos(chassis_angle))
            ldy = wheel_oval_coords[1] + R * self.scale * (1 - 0.5 * math.sin(chassis_angle))
            lux = wheel_oval_coords[0] + R * self.scale * (1 - 0.5 * math.cos(chassis_angle)) + l * self.scale * math.sin(chassis_angle)
            luy = wheel_oval_coords[1] + R * self.scale * (1 - 0.5 * math.sin(chassis_angle)) - l * self.scale * math.cos(chassis_angle)
            rux = wheel_oval_coords[0] + R * self.scale * (1 + 0.5 * math.cos(chassis_angle)) + l * self.scale * math.sin(chassis_angle)
            ruy = wheel_oval_coords[1] + R * self.scale * (1 + 0.5 * math.sin(chassis_angle)) - l * self.scale * math.cos(chassis_angle)
            rdx = wheel_oval_coords[0] + R * self.scale * (1 + 0.5 * math.cos(chassis_angle))
            rdy = wheel_oval_coords[1] + R * self.scale * (1 + 0.5 * math.sin(chassis_angle))
            self.model_canvas.coords(self.chassis_rect, ldx, ldy, lux, luy, rux, ruy, rdx, rdy, ldx, ldy)
            self.model_canvas.after(10, self.model_canvas.update())

def Experiments(g):
    rs_i = RobotState(0, 0, -0.1, 0)
    rs_g = RobotState(0, 0, 0, 0)
    important = [False, False, True, False]
    i_i = GetBoundaryPointInd(g, rs_i, important)
    i_g = GetBoundaryPointInd(g, rs_g, important)
    if (i_i == -1) or (i_g == -1):
        print("not found states")
        return
    if (BFS(g, i_i, i_g)[0][i_g] == -1):
        print("not found path")
        return
    print(g[i_i][0])
    print(g[i_g][0])
    next_vertex = DijkstraAlgorithm(ReverseGraph(g), i_g, i_i)[0]
    Q = QLearning(g, i_i, i_g)
    sum_tau_d = []
    sum_tau_q = []
    for k in range(1000):
        i_cur = i_i
        sum_tau = 0
        found = True
        while (i_cur != i_g):
            if (next_vertex[i_cur] == -1):
                print("not found path")
                found = False
                break
            e_cur = 0
            for i in range(len(g[i_cur][1])):
                if (g[i_cur][1][i][0] == next_vertex[i_cur]):
                    e_cur = i
                    break
            sum_tau += g[i_cur][1][e_cur][1]
            is_brake, e_cur = GetChangedAction(g, i_cur, e_cur)
            i_cur = g[i_cur][1][e_cur][0]
            #print(g[i_cur][0], is_brake)
        #print(sum_tau)
        if (found):
            sum_tau_d.append(sum_tau)
        i_cur = i_i
        sum_tau = 0
        found = True
        while (i_cur != i_g):
            if (BFS(g, i_cur, i_g)[0][i_g] == -1):
                print("not found path")
                found = False
                break
            e_cur = GetAction(Q, i_cur, -1)
            sum_tau += g[i_cur][1][e_cur][1]
            is_brake, e_cur = GetChangedAction(g, i_cur, e_cur)
            i_cur = g[i_cur][1][e_cur][0]
            #print(g[i_cur][0], is_brake)
        #print(sum_tau)
        if (found):
            sum_tau_q.append(sum_tau)
    print(sum_tau_d)
    print(sum_tau_q)
    print(sum(sum_tau_d) / len(sum_tau_d))
    print(sum(sum_tau_q) / len(sum_tau_q))

graph = BuildGraph(RobotState(0, 0, -0.1, 0))
for i in range(len(graph)):
    print(graph[i][0])
#chosen_method = DijkstraAlgorithm #Убрать комментарий с одной из следующих трех строк, чтобы выбрать метод построения пути
#chosen_method = BFS
#chosen_method = QLearning
#gm = GraphicalModel() #Запуск графической модели. При каких-то ошибках программа выдает текст в консоли
Experiments(graph) #Запуск сравнения методов построения пути