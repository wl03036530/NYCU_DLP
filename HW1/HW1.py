import numpy as np
import matplotlib.pyplot as plt
import random

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)

def generate_linear(n = 100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []

    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x ,y ,pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
        
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.show()

def forward(x, w1, w2, w3):
    Z1 = sigmoid(np.dot(x, w1))
    Z2 = sigmoid(np.dot(Z1, w2))
    Z3 = sigmoid(np.dot(Z2, w3))

    # without activation function
    #Z1 = np.dot(x, w1)
    #Z2 = np.dot(Z1, w2)
    #Z3 = np.dot(Z2, w3)

    return Z1, Z2, Z3

def backward(x, Z1, Z2, y_pred, y, w1, w2, w3, learning_rate):
    tempW3 = (y_pred - y) * derivative_sigmoid(y_pred)
    tempW2 = np.dot(tempW3, w3.T) * derivative_sigmoid(Z2)
    tempW1 = np.dot(tempW2, w2.T) * derivative_sigmoid(Z1)

    # without activation function
    #tempW3 = (y_pred - y)
    #tempW2 = np.dot(tempW3, w3.T)
    #tempW1 = np.dot(tempW2, w2.T)

    delta_w3 = np.dot(Z2.T, tempW3)
    delta_w2 = np.dot(Z1.T, tempW2)
    delta_w1 = np.dot(x.T, tempW1)

    #delta_w3 = np.dot(Z2.T, derivative_sigmoid(y) * (y_pred - y))
    #delta_w2 = np.dot(Z1.T, np.dot((y_pred - y)*derivative_sigmoid(y),  w3.T)*derivative_sigmoid(Z2))
    #delta_w1 = np.dot(x.T,  np.dot(np.dot((y_pred - y)*derivative_sigmoid(y),  w3.T)*derivative_sigmoid(Z2), w2.T)*derivative_sigmoid(Z1))
    #delta_w1 = derivative_sigmoid(Z1)*x.T * derivative_sigmoid(Z2)*w2.T * derivative_sigmoid(y)*w3.T * (y_pred - y)
    
    return w1 - learning_rate*delta_w1, w2 - learning_rate*delta_w2, w3 - learning_rate*delta_w3

if __name__ == '__main__':
    # x is coord(x, y), y is label(ground truth)
    # linear
    #x, y = generate_linear(n = 100)

    # XOR
    x, y = generate_XOR_easy()

    # initial weight(w1, w2, w3)
    hidden_unit_num = 30
    w1 = np.random.normal(0, 0.5, (2, hidden_unit_num))
    w2 = np.random.normal(0, 0.5, (hidden_unit_num, hidden_unit_num))
    w3 = np.random.normal(0, 0.5, (hidden_unit_num, 1))

    # train
    print('start training')
    learning_rate = 1
    epochs = 1000
    loss = []
    for e in range(epochs):
        for i in range(x.shape[0]):
            input = np.reshape(x[i], (1,2))
            Z1, Z2, y_pred = forward(input, w1, w2 ,w3)
            #MSE = (1/2) * np.sum((y_pred - y[i]) * (y_pred - y[i]))
            #loss.append(MSE)
            w1, w2, w3 = backward(input, Z1, Z2, y_pred, y[i], w1, w2, w3, learning_rate)
            
        MSE = 0
        for i in range(x.shape[0]):
            input = np.reshape(x[i], (1,2))
            Z1, Z2, y_pred = forward(input, w1, w2 ,w3)
            MSE += (1/2) * np.sum((y_pred - y[i]) * (y_pred - y[i]))

        MSE /= x.shape[0]
        loss.append(MSE)
        if e % 100 == 0:
            print (f'Epoch: {e}, loss: {MSE: 5.9f}')
    
    #test
    print('start testing')
    Z1, Z2, y_pred = forward(x, w1, w2 ,w3)
    print(y_pred)
    y_pred = np.round(y_pred)
    error = 0
    for i, j in zip(y_pred, y):
            if i != j:
                error += 1

    show_result(x, y, y_pred)
    plt.plot(loss)
    plt.show()
    print("Accuracy: %.2f" % ((1 - (error / x.shape[0])) * 100) + '%')


    