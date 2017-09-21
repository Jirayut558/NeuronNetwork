import pprint
import math
import operator
import random
from functools import reduce
import codecs

w_ = []
wt_ = []
bias_ = []
biast_ = []

testoutput_ = []

learningRate = 0.1
momentumRate = 0.1

def main():
    global w_,wt_,bias_,biast_
    print("Start...")

#-------Seting parameter -------
    layer = [4,6,3]
    initial_weight(layer)
    input_file = "iris.pat"
    input_test = "/Users/jirayutk./Project/projectfile/pretrainset/testset.csv"
#------setting input data--------
    x, y, d, e, g = initial_parameter(layer)


# ------train propagation--------
    if "trainset" in input_file:
        inputfile = codecs.open (input_file,'r','utf-8')
        testfile = codecs.open(input_test, 'r', 'utf-8')
        c = 0
        dataset = []

        for line in inputfile:
            if (c==0):
                c+=1
                continue
            try:
                data = line.split(",")
                output = data[4].replace("\r","")
                dataset.append([float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(output)])
            except:
                continue
        testset = []
        for line in testfile:
            if (c==0):
                c+=1
                continue
            try:
                data = line.split(",")
                output = data[4].replace("\r","")
                testset.append([float(data[0]),float(data[1]),float(data[2]),float(data[3]),float(output)])
            except:
                continue
        Es = 10
        while(Es>=0.12):
            tmp_E = 0
            for data in dataset:
                x, y, d, e, g = initial_parameter(layer)
                x = input_setting(data[:-1], x)
                if data[4:][0] == 0.0:
                    d = [1, 0]
                elif data[4:][0] == 1.0:
                    d = [0, 1]
                y, e, E = feedforword(x, y, d, e)
                backpropagation(x, y, e, g)
                tmp_E += E

            Es = tmp_E / len(dataset)
            print(Es)
        # -----------Test data for validation -----------------------#
        true_count = 0
        false_count = 0
        num = 0
        initial_testoutput(d)
        for data in testset:
            x = input_setting(data[:-1], x)
            if data[4:][0] == 0.0:
                d = [1, 0]
            elif data[4:][0] == 1.0:
                d = [0, 1]

            isTrue = test_data(x, d, y, e)
            if isTrue:
                true_count += 1
            else:
                false_count += 1
            num += 1
        print((true_count / (true_count + false_count)) * 100)
        pprint.pprint(testoutput_)

    if "iris" in input_file:
        print("IRIS")
        dataset = predata_iris(input_file)

        #--------------- 10% cross validation data ------------#
        for i in range(len(dataset[0])):
            testset = []
            trainset = []
            initial_weight(layer)
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
            Es = 10
            k = 0
            while(Es>=0.01):
                tmp_E = 0
                for data in trainset:
                    x, y, d, e, g = initial_parameter(layer)
                    x = input_setting(data[:-1],x)
                    if data[4:][0] == 1.0:
                        d = [1,0,0]
                    elif data[4:][0] == 2.0:
                        d = [0,1,0]
                    elif data[4:][0] == 3.0:
                        d = [0,0,1]
                    y, e, E = feedforword(x, y, d, e)
                    backpropagation(x, y, e, g)
                    tmp_E += E
                k+=1
                Es = tmp_E/len(trainset)
                print(Es)
            print(k)
            fileoutput = open("/Users/jirayutk./Project/CI/w/w_output.txt",'a')
            fileoutput.write("-------------------------------------------\n")
            for ii in range(len(w_)):
                fileoutput.write(str(ii)+"\n")
                for jj in range(len(w_[ii])):
                    fileoutput.write(str(jj) + "\n")
                    for kk in range(len(w_[ii][jj])):
                        fileoutput.write(str(w_[ii][jj][kk])+"\n")

            fileoutput.close()
            # -----------Test data for validation -----------------------#
            true_count = 0
            false_count = 0
            num = 0
            initial_testoutput(d)
            for data in testset:
                x = input_setting(data[:-1], x)
                if data[4:][0] == 1.0:
                    d = [1, 0, 0]
                elif data[4:][0] == 2.0:
                    d = [0, 1, 0]
                elif data[4:][0] == 3.0:
                    d = [0, 0, 1]
                isTrue = test_data(x, d, y, e)
                if isTrue:
                    true_count += 1
                else:
                    false_count += 1
                num += 1
            print((true_count/(true_count+false_count))*100)
            pprint.pprint(testoutput_)

            # -----------Test data for validation -----------------------#
            true_count = 0
            false_count = 0
            num = 0
            initial_testoutput(d)
            for data in trainset:
                x = input_setting(data[:-1], x)
                if data[4:][0] == 1.0:
                    d = [1, 0, 0]
                elif data[4:][0] == 2.0:
                    d = [0, 1, 0]
                elif data[4:][0] == 3.0:
                    d = [0, 0, 1]
                isTrue = test_data(x, d, y, e)
                if isTrue:
                    true_count += 1
                else:
                    false_count += 1
                num += 1
            print((true_count / (true_count + false_count)) * 100)
            pprint.pprint(testoutput_)
            print("------------")
    if "cross" in input_file:
        print("CROSS")
        testset,trainset = predata_cross(input_file)
        for k in range(len(trainset)):
            random.shuffle(trainset)
            initial_weight(layer)
            count = 0
            Es = 10
            ep = 0
            while(Es>=0.11):
                tmp_E = 0
                for i in trainset[k]:
                    x, y, d, e, g = initial_parameter(layer)
                    x=input_setting(i[:-1],x)
                    d=i[-1:][0]
                    y, e,E = feedforword(x,y,d,e)
                    backpropagation(x,y,e,g)
                    tmp_E += E
                ep+=1
                Es = tmp_E/len(trainset[k])
            print(ep)
            # -----------Test data for validation -----------------------#
            true_count = 0
            false_count = 0
            num = 0
            initial_testoutput(d)
            for data in testset[k]:
                x = input_setting(data[:-1], x)
                d = data[2]
                isTrue = test_data(x, d, y, e)
                if isTrue:
                    true_count += 1
                else:
                    false_count += 1
                num += 1
            print((true_count / (true_count + false_count)) * 100)
            pprint.pprint(testoutput_)
            # ---------- test data for trainset -------
            true_count = 0
            false_count = 0
            num = 0
            initial_testoutput(d)
            for data in trainset[k]:
                x = input_setting(data[:-1], x)
                d = data[2]
                isTrue = test_data(x, d, y, e)
                if isTrue:
                    true_count += 1
                else:
                    false_count += 1
                num += 1
            print((true_count / (true_count + false_count)) * 100)
            pprint.pprint(testoutput_)

            print("----------------")

'''
Test Data
'''
def output_analysis(y):
    for i in range(len(y[len(y)-1])):
        if y[len(y)-1][i] < 0.5:
            y[len(y) - 1][i] = 0
        elif y[len(y)-1][i] > 0.5:
            y[len(y) - 1][i] = 1
        else:
            y[len(y) - 1][i] = random.randint(0,1)
    return y
def initial_testoutput(d):
    # testdata[Expect][Actual]
    global testoutput_
    testoutput_.clear()
    for i in range(len(d)):
        temp = []
        for j in range(len(d)):
            temp.append(0)
        testoutput_.append(temp)

def test_data(x, d, y, e):
    y , e, E = feedforword(x, y, d, e)
    y = output_analysis(y)
    istrue = 0
    check = False
    expect = d.index(1)
    try:
        actual = y[len(y)-1].index(1)
    except:
        actual = random.randint(0,len(d)-1)
    testoutput_[expect][actual]+=1

    for i in range(len(d)):
        if d[i] == y[len(y)-1][i]:
            istrue+=1
    if istrue == len(d):
        return True
    else:
        return False

'''
Initial Data
'''
def input_setting(input,in_x):
    for i in range(len(in_x)):
        in_x[i] = input[i]
    return in_x

def initial_weight(layer):
    global w_,wt_
    global bias_,biast_
    w_.clear()
    bias_.clear()
    biast_.clear()
    wt_.clear()
    for i in range(len(layer)-1):
        tmp1 =[]
        tmp11 = []
        for j in range(layer[i]):
            tmp2 = []
            tmp22 =[]
            for k in range(layer[i+1]):
                rand = random.uniform(-1.0, 1.0)
                if (rand == 0):
                    k -= 1
                else:
                    tmp2.append(rand)
                    tmp22.append(0)
            tmp1.append(tmp2)
            tmp11.append(tmp22)
        w_.append(tmp1)
        wt_.append(tmp11)
    for i in range(len(layer)-1):
        i+=1
        tmp = []
        tmp2 = []
        for j in range(layer[i]):
            rand = random.uniform(-1.0, 1.0)
            if (rand == 0):
                j -= 1
            else:
                tmp.append(rand)
                tmp2.append(0)
        bias_.append(tmp)
        biast_.append(tmp2)
def initial_parameter(layer):
    x = []
    y = []
    g = []
    e = []
    d = []
    for i in range(layer[0]):
        x.append(0)
    for i in range(len(layer)):
        if i!=0:
            tmp = []
            tmp2 = []
            for j in range(layer[i]):
                if i == len(layer)-1:
                    e.append(0)
                    d.append(0)
                tmp.append(0)
                tmp2.append(0)
            y.append(tmp)
            g.append(tmp2)

    return x,y,d,e,g

def activation_function(v):
    return 1 / (1 + math.pow(math.e, -(v)))
def derivative_function(v):
    return v*(1-v)

'''
Feedforward
'''
def feedforword(x,y,d,e):
    y = calcualte_y(x,y)
    e = calculate_error(y,d,e)

    tmp_e = 0
    for i in e:
        tmp_e = tmp_e + math.pow(i,2)
    E = tmp_e/2
    return y,e,E
def calculate_error(y,d,e):
    for i in range(len(y[len(y)-1])):
        e[i] = d[i] - y[len(y)-1][i]
    return e
def calcualte_y(x,y):
    global w_,bias_
    for i in range(len(y)):
        for j in range(len(y[i])):
            tmp_y=0
            if(i==0):
                for k in range(len(x)):
                    tmp_y = tmp_y + (w_[i][k][j]*x[k])
            else:
                for k in range(len(y[i-1])):
                    tmp_y = tmp_y + (w_[i][k][j]*y[i-1][k])
            tmp_y = tmp_y + bias_[i][j]
            y[i][j] = activation_function(tmp_y)
    return y

'''
Backpropagation
'''
def backpropagation(x,y,e,g):
    g = GradianUpdate(y,e,g)
    WeightUpdate(x,y,g)

def GradianUpdate(y,e,g):
    global w_
    # ----output gradian-------
    for i in range(len(y[len(y) - 1])):
        g[len(y) - 1][i] = e[i] * derivative_function(y[len(y) - 1][i])

    # ----hidden gradian-------
    for i in range(len(y)):
        n = len(y) - 2 - i
        if n < 0:
            continue
        for j in range(len(y[n])):
            tmp_g = 0
            for k in range(len(y[n + 1])):
                tmp_g = tmp_g + (g[n + 1][k] * w_[n + 1][j][k])
            g[n][j] = derivative_function(y[n][j]) * tmp_g
    return g
def WeightUpdate(x,y,g):
    global w_,bias_,wt_,biast_
    for i in range (len(w_)):
        for j in range (len(w_[i])):
            tmp_w = 0
            for k in range(len(w_[i][j])):
                if i==0:
                    tmp_w = (momentumRate * (w_[i][j][k] - wt_[i][j][k])) + (learningRate*x[j]*g[i][k])
                else:
                    tmp_w = (momentumRate*(w_[i][j][k]-wt_[i][j][k])) + ( learningRate*y[i-1][j]*g[i][k])
                wt_[i][j][k] = w_[i][j][k]
                w_[i][j][k] = w_[i][j][k] + tmp_w
    for i in range(len(bias_)):
        for j in range(len(bias_[i])):
            tmp_bias = (momentumRate * (bias_[i][j] - biast_[i][j])) + (learningRate * g[i][j])
            biast_[i][j] = bias_[i][j]
            bias_[i][j] = bias_[i][j] + tmp_bias
'''
Predata
'''
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
    #------------10% cross validation data ----------
    testset = []
    trainset = []
    for i in range(10):
        testset.append(dataset[i:i+20])
        test = dataset[i:i+20]
        trainset.append([x for x in dataset if (x not in test)])
    return testset,trainset
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
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__ == '__main__':
    main()