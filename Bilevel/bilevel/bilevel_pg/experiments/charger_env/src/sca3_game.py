

import random
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt 
from scipy.optimize import fsolve, root
import itertools
import json
import multiprocessing


# 四叉树加速选点 (not implemented)
from QuadTree import QuadTreeNode

# 随机种子
seed = 10
random.seed(seed)

# 充电器总量
total_charger = 5
total_sensor  = 20
# 图大小
map_width     = 500.0 # width -> x
map_height    = 500.0 # height -> y
# 充电器波长
c_lambda = 32.8
# 充电器相关常数
c_alpha = 10.9
c_beta = 9.9
# 充电器近距离无辐射
c_exemption = 0.5 * c_lambda
# 充电器最小充电阙值 (mw)
c_min_mw = 0.1
# 充电器最大充电距离
c_max_d = np.sqrt(c_alpha * 1000 / c_min_mw) - c_beta
# 最大充电距离等效波长倍数 (ceil)
c_max_lambda = int((c_max_d // c_lambda))
# 充电器辐射离散化 (w))
c_gap = 0.001

# sensor最大充电功率
p_max = 5.0
# 
p_0 = 3000.0

# 振幅
A = 1

# 充电器离散化近似距离
gap = 1.0
def approx(dis):
    offset = dis % gap
    dis += (gap-offset) if offset > (dis/2) else -offset
    return dis

def getRadiation(charger_idx, node):
    d = EulerDis(c_positions[charger_idx], node)
    return c_alpha / (d + c_beta)**2

def getIFRadiation(node):
    """
    get radiation from charger[charger_idx] -> node \\
    return the radiation \\
    O(c^2)
    """
    p0 = []
    phi = []
    d = []
    for j in range(total_charger):
        if not c_status[j]: 
            d.append(0)
            phi.append(0)
            p0.append(0)
        else:
            d.append(EulerDis(c_positions[j], node))
            phi.append(2*np.pi + 2*np.pi*d[j]/c_lambda)
            p0.append(p_0 * c_alpha / (d[j] + c_beta)**2)
    r = 0
    for j in range(total_charger):
        if not c_status[j]: continue
        r += p0[j]
        p_intf = 0
        for k in range(total_charger):
            if not c_status[k] or k == j: continue
            p_intf += np.sqrt(p0[j]*p0[k])*np.cos(phi[j]-phi[k]);
        r += p_intf
    return r

# 初始化地图
Map   = [[]]

# 随机部署充电器
def generateRandomChargers(s, t):
    chargers = []
    if(s >= t): return chargers
    for i in range(s, t):
        charger = {
            "id": i,
            "status" : False,
            "x" : random.uniform(0, map_width ),
            "y" : random.uniform(0, map_height),
        }
        chargers.append(charger)
    return chargers

# 随机部署传感器
def generateRandomSensors(s, t):
    sensors = []
    if(s >= t): return sensors
    for i in range(s, t):
        sensor = {
            "id": i,
            "x" : random.uniform(0, map_width ),
            "y" : random.uniform(0, map_height),
        }
        sensors.append(sensor)
    return sensors

# 读取现有的地图数据
try: 
    with open('map.json', 'r') as file:
        loaded_data = json.load(file)
        origin_charger = loaded_data["MapInfo"]["total_charger"]
        origin_sensor  = loaded_data["MapInfo"]["total_sensor"]
        origin_mapinfo = loaded_data["MapInfo"]
except FileNotFoundError: 
    intersection_flag = True
    origin_charger = 0
    origin_sensor  = 0

print("read chargers:", origin_charger)

# find intersections: O(n^2logN)
map_x = np.arange(0, map_width + gap, gap)
map_y = np.arange(0, map_height + gap, gap)
c_intersections = [np.array([]) for _ in range(total_charger)]
i_radiation = []
intersection_pts = []
intersection_flag = False

# 生成.json文件保存地图信息
MapInfo = {
    "total_charger" : total_charger,
    "total_sensor"  : total_sensor,
    "map_width"     : map_width,
    "map_height"    : map_height,
    "map_seed"      : seed,
}
def write_json():
    JsonData = {
        "MapInfo"       : MapInfo,
        "Chargers"      : chargers,
        "Sensors"       : sensors,
        "Intersections" : intersection_pts
    }
    with open('map.json', 'w') as file:
        json.dump(JsonData, file, indent = 4)

# if(origin_charger == 0 or origin_sensor == 0 or
# origin_mapinfo["map_width" ] != MapInfo["map_width" ] or
# origin_mapinfo["map_height"] != MapInfo["map_height"] or
# origin_mapinfo["map_seed"  ] != MapInfo["map_seed"]   or
# len(loaded_data["Intersections"]) == 0
#     ):
if origin_mapinfo != MapInfo:
    intersection_flag = True
    origin_charger = 0
    origin_sensor  = 0
chargers = generateRandomChargers(origin_charger, total_charger)
sensors  = generateRandomSensors (origin_sensor , total_sensor )
if origin_sensor: sensors  = loaded_data["Sensors" ] + sensors
if origin_charger: chargers = loaded_data["Chargers"] + chargers
if not intersection_flag: intersection_pts = loaded_data["Intersections"]

# 写入.json文件，保存地图信息
write_json()

EulerDis = lambda x, y: np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2))
def EulerDis2(ch, vars):
    return sp.sqrt((ch[0] - vars[0])**2 + (ch[1] - vars[1])**2)

c_positions = np.array([(charger['x'], charger['y']) for charger in chargers])
c_status = np.array([charger['status'] for charger in chargers])
s_positions = np.array([(sensor['x'], sensor['y']) for sensor in sensors])

# 通过bool类型数列修改充电器的状态
def changeChargerMode(switch_sequence):
    global c_status
    c_status = switch_sequence
    for status, charger in zip(switch_sequence, chargers):
        charger["status"] = status
    write_json()

# 计算平均传感器充电效用
def chargingEffiency(switch_sequence): 
    changeChargerMode(switch_sequence)
    p_sum = 0
    for i in range(total_sensor):
        p0 = []
        phi = []
        d = []
        for j in range(total_charger):
            if not c_status[j]: 
                d.append(0)
                phi.append(0)
                p0.append(0)
            else:
                d.append(EulerDis(c_positions[j], s_positions[i]))
                phi.append(2*np.pi + 2*np.pi*d[j]/c_lambda)
                p0.append(p_0 * c_alpha / (d[j] + c_beta)**2)
        p = 0
        for j in range(total_charger):
            if not c_status[j]: continue
            p += p0[j]
            p_intf = 0
            for k in range(total_charger):
                if not c_status[k] or k == j: continue
                p_intf += np.sqrt(p0[j]*p0[k])*np.cos(phi[j]-phi[k]);
            p += p_intf
            if p > p_max: 
                p = p_max
                break
        p_sum += p / p_max
    p_ava = p_sum / total_sensor
    # print("average charging efficiency", p_ava)
    return p_ava




# 获取单个节点的辐射值
def getNodeRadiation(node):
    r = 0
    for i in range(total_charger):
        if not c_status[i]: continue
        d = EulerDis(c_positions[i], node)
        r += p_0 * c_alpha / (d + c_beta)**2
    return r

# 获取所有交点的辐射值
def getTotalRadiation():
    max_r = 0
    for i in intersection_pts:
        r = getIFRadiation(i)
        max_r = max(max_r, r)
    return max_r
        

# 遍历节点计算双曲线的方式获取交点
def findIntersections():

    def hyperbola_eqs(vars, i, ch1, ch2, ch3, ch4):
        eq1 = EulerDis(ch1, vars) - EulerDis(ch2, vars) - i * c_lambda
        eq2 = EulerDis(ch3, vars) - EulerDis(ch4, vars) - i * c_lambda
        return [eq1, eq2]

    for k in range(-c_max_lambda, c_max_lambda+1):
        if k == 0: continue
        print(k)
        for (c1, c2), (c3, c4) in itertools.product(
            itertools.combinations(range(total_charger), 2),
            repeat=2):
            if (c_status[c1] == False or c_status[c2] == False or c_status[c3] == False or c_status[c4] == False): 
                continue
            if c1 == c3 and c2 == c4:
                continue
            print(c1, c2, c3, c4)
            a1,b1,c1,d1,e1 = c_positions[c1][0], c_positions[c1][1], c_positions[c3][0], c_positions[c3][1], k * c_lambda
            a2,b2,c2,d2,e2 = c_positions[c3][0], c_positions[c3][1], c_positions[c4][0], c_positions[c4][1], k * c_lambda
            print("eq1 args",a1,b1,c1,d1,e1)
            print("eq2 args",a2,b2,c2,d2,e2)
            initial_guess = [map_width/2, map_height/2]
            def equations(vars, a1, b1, c1, d1, e1, a2, b2, c2, d2, e2):
                x, y = vars
                eq1 = np.sqrt((a1 - x)**2 + (b1 - y)**2) - np.sqrt((c1 - x)**2 + (d1 - y)**2) - e1
                eq2 = np.sqrt((a2 - x)**2 + (b2 - y)**2) - np.sqrt((c2 - x)**2 + (d2 - y)**2) - e2
                return [eq1, eq2]
            solution = fsolve(equations, initial_guess, args=(a1, b1, c1, d1, e1, a2, b2, c2, d2, e2))
            solutions = root(equations, initial_guess, args=(a1, b1, c1, d1, e1, a2, b2, c2, d2, e2), method='hybr')
            print(solution)
            print(equations(solution, a1, b1, c1, d1, e1, a2, b2, c2, d2, e2 ))
            print(solutions.x)
            # real_solution = [sol for sol in solution if all(val.is_real for val in sol)]
            # print(real_solution)
            """
            idx = len(intersection_pts)
            print("append intersection: ", solution, idx)
            c_intersections[c1].append(idx)
            c_intersections[c2].append(idx)
            if c3 != c1 and c3 != c2: c_intersections[c3].append(idx)
            if c4 != c1 and c4 != c2: c_intersections[c4].append(idx)
            intersection_pts.append(solution)
            i_radiation.append(0)
            """

# 遍历地图每一点的方式获取交点
def intersectionTraverse(): 
    print("get intersection:", c_max_d)
    cnt = 0
    for k in range(-c_max_lambda, c_max_lambda+1):
        print("lambda =", k)
        for (i, x), (j, y) in itertools.product(enumerate(map_x), enumerate(map_y)):
            z_sum = 0
            flag = False
            cnt = 0
            # print(i,j,x,y)
            for c1, c2 in itertools.combinations(range(total_charger), 2):
                if (c_status[c1] == False or c_status[c2] == False): 
                    continue
                dis1 = EulerDis(c_positions[c1], (x, y))
                dis2 = EulerDis(c_positions[c2], (x, y))
                if dis1 > c_max_d or dis2 > c_max_d or dis1 < c_exemption or dis2 < c_exemption:
                    continue
                z = dis1 - dis2 - k * c_lambda
                z_sum += z
                if np.abs(z) < 0.01:
                    # print(c1, c2, "charging", i, j, "for lambda", k, z, dis1, dis2)
                    flag = True
                    break
                elif np.abs(z) < 0.1:
                    cnt += 1
            if flag or cnt > 1:
                print("ans", i, j)
                intersection_pts.append((x,y))
                i_radiation.append(0)
    
    print(len(intersection_pts), "intersections init completed!")


def worker(args):
    cnt = 0
    k = args[0]
    worker_intersection_pts = []
    worker_i_radiation = []
    print("lambda =", k)
    for (i, x), (j, y) in itertools.product(enumerate(map_x), enumerate(map_y)):
        z_sum = 0
        flag = False
        # print(i,j,x,y)
        for c1, c2 in itertools.combinations(range(total_charger), 2):
            if (c_status[c1] == False or c_status[c2] == False): 
                continue
            dis1 = EulerDis(c_positions[c1], (x, y))
            dis2 = EulerDis(c_positions[c2], (x, y))
            if dis1 > c_max_d or dis2 > c_max_d or dis1 < c_exemption or dis2 < c_exemption:
                continue
            z = dis1 - dis2 - k * c_lambda
            z_sum += z
            if np.abs(z) < 0.01:
                # print(c1, c2, "charging", i, j, "for lambda", k, z, dis1, dis2)
                flag = True
                break
        if flag:
            # print("ans", i, j, k)
            worker_intersection_pts.append((x,y))
            worker_i_radiation.append(0)
    return worker_intersection_pts, worker_i_radiation

def intersectionTraverseThread():
    num_processes = multiprocessing.cpu_count()
    processes = []
    for k in range(-c_max_lambda, c_max_lambda+1):
        p = multiprocessing.Process(target=worker, args=(k, ))
        processes.append(p)
        if(len(processes) < num_processes):
            p.start()
        else:
            for p in processes:
                p.join()
    
    for p in processes:
        p.join()



def getMaxRadiation(switch_sequence):
    changeChargerMode(switch_sequence)
    global intersection_flag
    if intersection_flag:
        intersectionTraverse()
        # intersectionTraverseThread()
        intersection_flag = False
        write_json()
    max_r = getTotalRadiation()
    return max_r

def sca3_step(s):
    return getMaxRadiation(s), chargingEffiency(s)
