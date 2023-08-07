from cmath import sqrt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


def prepare_data(data , days): #data = outputs
    res = []
    inputs = []
    outputs = []
    for i in range(max(days), len(data)):
        outputs.append(data[i])
        line = []
        for d in days:
            line.append(data[i - d][0])
        inputs.append(line)
        #res.append([data[i],data[i -1],data[i - 365]]) #donnée de la veille n et n-1
    return inputs,outputs

def read_data(filename):
    f = open(filename,'r')
    f.readline() #skip first line

    inputs = []
    outputs = []
    for line in f.readlines():
        fields = line.split(';')
        inputs.append([float(fields[1])])
        outputs.append([float(fields[2].replace(',','.'))])


    f.close()
    return inputs,outputs


if __name__ == "__main__":
    inputs, outputs = read_data('gasprice.csv')
    inputs, outputs = prepare_data(outputs,[1,2, 3])

    # inputs = prices[:,0]
    # outputs = prices[:,1:]
    

   #display first 1à values
    for i in range(10):
        print(inputs[i],outputs[i]);
   
    
    #display first 1à values
    # for i in range(10):
    #     print(inputs[i],'->',outputs[i]);

    regression = LinearRegression()
    regression.fit(inputs,outputs)

    print(f'Y = {regression.coef_} * x + {regression.intercept_}')
    
    points = []
    points.append([inputs[0], regression.predict([inputs[0]])])
    points.append([inputs[-1], regression.predict([inputs[-1]])])
    print(points)


    rms = 0
    for i in range(len(inputs)):
       err = outputs[i][0] - regression.predict([inputs[i]])[0]
       err *= err
       rms += err * err / len(inputs)

    rms = sqrt(rms)

   # datetime.fromisoformat('2020-')

    future = datetime.fromisoformat('2024-05-10T10:00:20').timestamp()
    prediciton = regression.predict([[future]])[0]
    print(prediciton)

    print(f'Roor mean square error: {rms}')

    plt.plot(inputs,outputs) #    plt.plot(inputs,outputs,'ro')
    plt.plot([inputs[0][0], inputs[-1][0]], [regression.predict([inputs[0]])[0],regression.predict([inputs[-1]])[0]])  #pemier timestamp  deuxieme time / prediction de ma courbe du point 0 / predicion de ma courbe du point B

    plt.plot([inputs[-1][0],future],
                [regression.predict([inputs[-1]])[0][0],
               prediciton[0]])
    plt.show()
