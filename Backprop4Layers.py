import math
import random

random.seed()
 
 # calculate a random number between a and b
def rand(a, b):
    return (b-a)*random.random() + a
 
def makeMtx(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        x=x%1
        return 1 / (1 + math.exp(-x))

def dsigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

class NeuralNet:
    def __init__(self, ni, nh, nh2, no):
        # number of input, hidden, hidden2 and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.nh2 = nh2
        self.no = no

        # number of times activations are done, since it's just a loop of sigmoid functions
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ah2 = [1.0]*self.nh2
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMtx(self.ni, self.nh)
        self.wh = makeMtx(self.nh,self.nh2)
        self.wo = makeMtx(self.nh2, self.no)
         # set them to random vaules
        for i in range(0,self.ni):
            for j in range(0,self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        
        for i in range(0,self.nh):
            for j in range(0,self.nh2):
                self.wh[i][j] = rand(-0.2, 0.2)
                
        for j in range(0,self.nh2):
            for k in range(0,self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum   
        self.ci = makeMtx(self.ni, self.nh)
        self.ch = makeMtx(self.nh, self.nh2)
        self.co = makeMtx(self.nh2, self.no)

    def feedforward(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('The input nodes should be the same length as the data input length. E.G. a point classifying system needs input nodes = dimension of the points in the training set ')

        # input nodes
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]

        # hidden node activations
        for j in range(self.nh):
            sumOf = 0.0
            for i in range(self.ni):
                sumOf = sumOf + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sumOf)
        
        # hidden2 node activations
        for j in range(self.nh2):
            sumOf = 0.0
            for i in range(self.nh):
                sumOf = sumOf + self.ah[i] * self.wh[i][j]
            self.ah2[j] = sigmoid(sumOf)
        # output node activations
        for k in range(self.no):
            sumOf = 0.0
            for j in range(self.nh2):
                sumOf = sumOf + self.ah2[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sumOf)
        #return guess
        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error
        
        # calculate error for hidden layer 2
        hidden_deltas2 = [0.0] * self.nh2
        for j in range(self.nh2):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas2[j] = dsigmoid(self.ah2[j]) * error
        
        # calculate error for hidden layer 1
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.nh2):
                error = error + hidden_deltas2[k]*self.wh[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error
         # update weights between hidden layer 1 and 2
        for j in range(self.nh2):
            for k in range(self.no):
                change = output_deltas[k]*self.ah2[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]
        
        # update weights between hidden layer 1 and hidden layer 2
        for j in range(self.nh):
            for k in range(self.nh2):
                change = hidden_deltas2[k]*self.ah[j]
                self.wh[j][k] = self.wh[j][k] + N*change + M*self.ch[j][k]
                self.ch[j][k] = change
        
        # update weights between input and hidden layer 1
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
       counter = 0
       for p in patterns:
           inputs = p[0]
           targets = p[1]
           guess = self.feedforward(inputs)
            
           if guess[0] >= .5 and targets[0] == 1:
                counter+=1
           elif guess[0] < .5 and targets[0] == 0:
                counter+=1  
       print((counter/len(patterns)) * 100, "% classification accuracy")
     
    def test2(self, patterns):
        print("1 means outside circle, 0 inside")
        for p in patterns:
            print(p[0], '->', self.feedforward(p[0]))          
         
             
 
    def Save_Weights(self):
        file_ = open('weights.txt', 'w')
        file_.write(self.wo, self.wh,self.wo)
        file_.close()
     

    def train(self, patterns, iterations=10, N=0.5, M=0.1):

        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedforward(inputs)
                error = error + self.backPropagate(targets, N, M)
            print("epoch number", i, 'error %-.5f' % error)
 
def demo():
    trainingset=[]
    radius = 2
    counter=0
    while counter < 500:
        point_x = rand(-5,5)
        point_y = rand(-5,5)
        if (point_x**2 + point_y**2) < radius**2:
            trainingset.append([[point_x,point_y],[0]])
        else:
            trainingset.append([[point_x,point_y],[1]])
        counter = counter + 1
    #define a network
    
     #test it with a randomly generated set of points inside or outside a cirlce
    testset=[]
    counter=0
    while counter < 100:
        point_x = rand(-5,5)
        point_y = rand(-5,5)
        if (point_x**2 + point_y**2) < radius**2:
            testset.append([[point_x,point_y],[0]])
        else:
            testset.append([[point_x,point_y],[1]])
        counter = counter + 1
    
     #Create a neural network with 2 hidden layers
    n = NeuralNet(2, 100, 100, 1)
    #test it with a randomly generated set of points inside or outside a cirlce
    #n.test2(testset)
    n.test(testset)
     # train it with a randomly generated set of points inside or outside a cirlce
    n.train(trainingset)
     # test it
    #n.test2(testset)
    n.test(testset)
demo()
