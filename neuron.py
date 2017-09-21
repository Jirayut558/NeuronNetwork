import math
import operator
import random
from functools import reduce

'''
x : input node
y : hidden and output node
w : weight
g : gradian
'''
delta_w = []
def main():
#######-----------input setting---------################
    input_layer = "2,2,2"
    input_file = "cross.pat"
    learnR =  0.2
    momentR = 0.1
#######-----------input setting---------################

    #initial parameter
    layer = input_layer.split(',')
    x, y, w, g,e = initial_data(layer)
    #d = [0]
    #x = set_x_value([0,1],x)
    #E, y, e = feedforward(x, d, y, w, e)
    #print (y)
    #w, g = backprobpagation(y, g, e, w, learnR, momentR)



    if "iris" in input_file:
        dataset = predata_iris(input_file)

        #--------------- 10% cross validation data ------------#
        k=0
        for i in range(len(dataset[0])):
            testset = []
            trainset = []
            for j in range(len(dataset[0][i])):
                testset.append(dataset[0][i][j])
                testset.append(dataset[1][i][j])
                testset.append(dataset[2][i][j])
                for k in range(len(dataset[0])):
                    if k!=i :
                        trainset.append(dataset[0][k][j])
                        trainset.append(dataset[1][k][j])
                        trainset.append(dataset[2][k][j])
            #------------ Train Data ------------------#
            random.shuffle(trainset)
            ii=0
            Es = 10
            while(Es>0.1):
                tmp_E = 0
                for data in trainset:
                    x = set_x_value(data[:-1],x)
                    if data[4:][0] == 1.0:
                        d = [1,0,0]
                    elif data[4:][0] == 2.0:
                        d = [0,1,0]
                    elif data[4:][0] == 3.0:
                        d = [0,0,1]
                    E, y, e = feedforward(x, d, y, w, e)
                    w, g = backprobpagation(y, g, e, w, learnR, momentR)
                    tmp_E += E
                    print (g)
                Es = tmp_E

                ii+=1
            #-----------Test data -----------------------#
            true_count = 0
            false_count = 0
            num = 0
            for data in testset:
                x = set_x_value(data[:-1],x)
                if data[4:][0] == 1.0:
                    d = [1,0,0]
                elif data[4:][0] == 2.0:
                    d = [0,1,0]
                elif data[4:][0] == 3.0:
                    d = [0,0,1]
                isTrue = test_data(x,d,y,w,e)
                if isTrue:
                    true_count+=1
                else:
                    false_count +=1
                num+=1
            print(true_count,false_count)
    if "cross" in input_file:
        print("CROSS")
        dataset = predata_cross(input_file)
        #trainset = dataset
        trainset = dataset[:-20]
        testset = dataset[179:]
        Es = 10
        while(Es>0.1):
            tmp_E = 0
            for i in trainset:
                x=set_x_value(i[:-1],x)
                d=i[-1:][0]
                E, y, e = feedforward(x, d, y, w, e)
                w, g = backprobpagation(y, g, e, w, learnR, momentR)
                tmp_E += E
                print(g)
            Es = tmp_E/len(trainset)

def test_data(x, d, y, w, e):
    E , y, e = feedforward(x,d,y,w,e)
    y = output_analysis(y)
    istrue = 0
    check = False

    for i in range(len(d)):
        if d[i] == y[len(y)-1][i]:
            istrue+=1
    if istrue == len(d):
        return True
    else:
        return False

'''
output analysis to 1 or -1
'''
def output_analysis(y):
    for i in range(len(y[len(y)-1])):
        if y[len(y)-1][i] < 0:
            y[len(y) - 1][i] = -1
        elif y[len(y)-1][i] > 0:
            y[len(y) - 1][i] = 1
        else:
            y[len(y) - 1][i] = random.randint(-1,1)
    return y
def predata_cross(pathfile):
    file = open(pathfile,'r')
    l = 0
    dataset = []
    tmp =[]
    for line in file:
        if l%3 == 1:
            tmp = []
            x= line.split("  ")
            tmp.append(float(x[0]))
            tmp.append(float(x[1]))
        if l%3 ==2:
            xx = line.split(" ")
            tmp.append([int(xx[0]),int(xx[1])])
        if l%3 == 0 and l!=0 :
            dataset.append(tmp)
        l+=1
    return (dataset)

def predata_iris(pathfile):
    file = open(pathfile,'r')
    data_list = [[],[],[]]
#seperate class of data
    i = 0
    for line in file:
        if i ==0:
            i+=1
            continue
        results = line.split("\t")
        results = [float(j) for j in results]

        if results[len(results) - 1] == 1:
            data_list[0].append(results)
        elif results[len(results) - 1] == 2:
            data_list[1].append(results)
        else:
            data_list[2].append(results)
        i+=1

    dataset = []
    for i in range(len(data_list)):
        s = int(0.1*len(data_list[i]))
        dataset.append(list(chunks(data_list[i],s)))
    return (dataset)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
'''
input:
    x : list of input
    d : Expected output
    y : Actual output
'''
def feedforward(x,d,y,w,e):
    y = calculate_y(y,x,w)
    e = calculate_error(e,d,y)
    tmp = 0
    for i in e:
        tmp+=math.pow(i,2)
    E = tmp/2
    return E,y,e

'''
input:
'''
def backprobpagation(y,g,e,w,LR,MR):
    g = FindGradian_output(g, e, y)
    g = FindGradian_hidden(y, g, w)
    w = updateWeight(w, g, y, LR, MR)
    return w,g

'''
e : sum(e(i))/2
e per data
'''
def calculate_error(e,d,y):
    for i in range(len(y[len(y)-1])-1):
        e[i] = (d[i] - y[len(y)-1][i])
    return e

def activation_fn(v):
    #return math.tanh(v/2)
    return 1/(1+math.pow(math.e,-v))

def derivative_fn(v):
    #exp1 = math.pow(math.e,(-v/2))
    #exp2 = math.pow(math.e,(v/2))
    #return (2)/((exp2+exp1)*(exp2+exp1))
    return v*(1-v)

def set_x_value(input,x):
    j = 0
    for i in range(len(x)):
        if j<len(x)-1:
            x[i] = input[j]
            j+=1
    return x

def initial_data(layer):
    y=[]
    w=[]
    x=[]
    g=[]
    for i in range(len(layer)):
        if i!=0:
            tmp_y = []
            for j in range(int(layer[i])+1):
                if j == int(layer[i]):
                    tmp_y.append(1)
                else:
                    tmp_y.append(0)
            y.append(tmp_y)
    for i in range(len(layer)):
        if i!=0:
            tmp_y = []
            for j in range(int(layer[i])+1):
                tmp_y.append(0)
            g.append(tmp_y)

    for i in range( len(y) ):
        tmp = []
        for j in range ( int(layer[i])+1):
            tmp_w = []
            for k in range (int(layer[i+1])+1):
                rand = random.uniform(-1.0,1.0)
                if (rand==0):
                    k-=1
                else:
                    tmp_w.append(rand)
            tmp.append(tmp_w)
        w.append(tmp)


    for i in range( len(y) ):
        tmp = []
        for j in range ( int(layer[i])+1):
            tmp_w = []
            for k in range (int(layer[i+1])+1):
                if (rand==0):
                    k-=1
                else:
                    tmp_w.append(0)
            tmp.append(tmp_w)
        delta_w.append(tmp)
    e = g[len(y) - 1][:-1]
    x = [0 for i in range(int(layer[0])+1)]
    x[int(layer[0])] = 1
    return x,y,w,g,e

def calculate_y(y,x,w):
    for i in range(len(y)):
        for j in range (len(y[i])):
            tmp = 0
            if i==0:
                for k in range(len(x)):
                    tmp+=(w[i][k][j])*(x[k])
            else:
                for k in range(len(y[i-1])):
                    tmp+=(w[i][k][j]*y[i-1][k])

            if (j!=len(y[i])-1):
                y[i][j] = activation_fn(tmp)
    return y

'''
input : e(n) and y(n)
output : g(n)
'''
def FindGradian_output(g,e,y):
    for i in range(len(e)):
        g[len(y)-1][i] = e[i]*derivative_fn(y[len(y)-1][i])
    return (g)
'''
input : learnR,y,g,w
output : g(update)
'''
def FindGradian_hidden(y,g,w):
    for i in range( len(y) ):
        n = (len(y)-2) - i
        if n<0:
            break
        for j in range( len(y[n]) ):
            tmp = 0
            for k in range ( len(y[n+1]) ):
                tmp+=g[n+1][k]*w[n+1][j][k]
            g[n][j] = derivative_fn(y[n][j])*tmp
    return g

'''
input : w,g,y,LR,MR
output : w(update)
'''
def updateWeight(w,g,y,LR,MR):
    global delta_w
    for i in range( len(w) ):
        for j in range (len(w[i])):
            tmp = 0
            for k in range (len(w[i][j])):
                tmp = MR*(w[i][j][k]-delta_w[i][j][k]) + LR*g[i][k]*y[i][k]
                w[i][j][k] = w[i][j][k] + tmp
                delta_w[i][j][k] = w[i][j][k]
    return w

if __name__ == '__main__':
    x = [1,2,3,4,5,6,7,8,9,10]

    tmp = (x[1:6])
    new_list = [x for x in dataset if (x not in tmp)]

    print(new_list,tmp)

