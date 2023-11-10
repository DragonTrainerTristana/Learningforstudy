import numpy as np
import matplotlib.pyplot as plt
import time

INF = 10**60 
N_INF = -10**60
DAMP = 0.3  
N_NODE = 17 
N_ITER = N_NODE*10
bLogSumExp = False
np.set_printoptions(precision=5)

def main():
    
    for i in range(N_NODE):
        points = np.random.uniform(0,100,(N_NODE,3))
        points[:,2] = -1200
        
    exemplars, exemplar1s = [-1 for _ in range(N_NODE)], [INF for _ in range(N_NODE)]
    s = [[i for i in range(N_NODE)] for _ in range(N_NODE)]
    a = [[0 for _ in range(N_NODE)] for _ in range(N_NODE)]
    r = [[0 for _ in range(N_NODE)] for _ in range(N_NODE)]
    
    s = update_similarity(s, points)
    
    for i in range(N_ITER):
        r = update_reli(a,s,r)
        a = update_avai(a,r)
        running = i
        exemplars,i = conclude_update(exemplars, exemplar1s, a, r, running)
   
    plot_results(exemplars, points)

def update_similarity(s ,points):
    for i in range(N_NODE):
        for j in range(N_NODE):
            s[i][j] = points[i][2]
        else :
            s[i][j] = -((points[i][1] - points[j][1]) ** 2) - (
                (points[i][0] - points[j][0]) ** 2
            )
    return s

def update_avai(a,r):
    
    for i in range(N_NODE):
        check = 0
        for j in range(N_NODE):
            if i == j:
                continue
            check += max(0, r[j][i])
        a[i][i] = check / 2 + a[i][i] / 2
        
    for i in range(N_NODE):
        for j in range(N_NODE):
            ret = 0
            if i == j:
                continue
            for k in range(N_NODE):
                if k == i or k == j:
                    continue
                ret += max(0, r[k][j])
            a[i][j] = min(0, r[j][j] / 2 + ret / 2) + a[i][j] / 2
    return a

def update_reli(a,s,r):
    for i in range(N_NODE):
        for j in range(N_NODE):
            maxi = N_INF
            for k in range(N_NODE):
                if k == j:
                    continue
                maxi = max(maxi, a[i][k] + s[i][k])
            r[i][j] = (s[i][j] - maxi) / 2 + r[i][j] / 2   
    return r

def conclude_update(exemplars, exemplar1s, a, r, running):
    for i in range(N_NODE):
        cur_exemplar, cur_value = i, N_INF
        for j in range(N_NODE):
            if a[i][j] + r[i][j] > cur_value:
                cur_exemplar = j
                cur_value = a[i][j] + r[i][j]
        exemplars[i] = cur_exemplar
    running += 1

    for i in range(N_NODE):
        if exemplars[i] != exemplar1s[i]:
            running = 0
            break
        
    for i in range(N_NODE):
        exemplar1s[i] = exemplars[i] 
        
    return exemplars, running   

def plot_results(exemplars, points):
    
    for i in range(len(exemplars)):
        print(i, exemplars[i])

    setofexe = list(set(exemplars))

    for i in range(len(setofexe)):
        x, y = [], []
        for j in range(len(exemplars)):
            if exemplars[j] == setofexe[i]:
                x.append(points[j][0])
                y.append(points[j][1])
        plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    main()