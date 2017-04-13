import math
import random

random.seed()

def rand(x, y):
    return (x-y)*random.random() + x

def createMatrix(I, J):
    mtx = []
    for i in range(I):
        mtx.append([0.0]*J)
    return mtx

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        x=x%1
        return 1 / (1 + math.exp(-x))

def dsigmoid(x):
    sigx = sigmoid(x)
    return sigx * (1 - sigx)

class NeuralNet:
    def __init__(self, numInputs,layers, numOutputs, LR):
        self.LR = LR
        self.numInputs = numInputs + 1
        self.numOutputs = numOutputs
        self.layers = [0.0] * len(layers)
        for i in range(0, len(layers)):
            self.layers[i] = layers[i]

        self.activations = [[]] * len(layers)
        self.deltas = [[]] *len(layers)
        for i in range(0,len(layers)):
            for j in range(0, int(layers[i])):
                self.activations[i].append(0.0)
                self.deltas[i].append(0.0)
        
        self.weights = [[]] * (len(layers) -1)
        for i in range(len(self.layers) -1):
            self.weights[i] = createMatrix(len(self.activations[i]),len(self.activations[i+1]))
            for j in range(len(self.activations[i])):
                for k in range(len(self.activations[i+1])):
                    self.weights[i][j][k] = rand(-0.2, 0.2)

    def feedforward(self, inputs):
        if len(inputs) != self.numInputs-1:
            raise ValueError('The input nodes should be the same length as the data input length. E.G. a point classifying system needs input nodes = dimension of the points in the training set ')

        self.activations[0] = inputs
        for i in range (1, len(self.layers)-1):
          self.activations[i+1] = self.calcLayer(self.activations[i],self.activations[i+1],self.weights[i])
      
        return self.activations[len(self.layers)-1]

    def calcLayer(self, activation1=[], activation2=[], weightsA1A2 =[[]]):
        for j in range(len(activation2)):
            sumOf = 0.0
            for i in range(len(activation1)):
                sumOf = sumOf + activation1[i] * weightsA1A2[i][j]
            activation2[j] = sigmoid(sumOf)
        return activation2


    def backPropagate(self, targets, N):
        if len(targets) != self.numOutputs:
            raise ValueError('wrong number of target values')

   
        output_deltas = [0.0] * self.numOutputs
        for k in range(self.numOutputs):
            error = targets[k]-self.activations[len(self.layers)-1][k]
            self.deltas[len(self.layers)-1][k] = dsigmoid(self.deltas[len(self.layers)-1][k]) * error
        
        for i in range((len(self.layers)-2),1,-1):
           self.deltas[i] = self.calcHDelta(self.activations[i],self.deltas[i],self.deltas[i-1], self.weights[i])
               
        for i in range((len(self.layers)-2),1,-1):
            self.weights[i] = self.weightUpdate(self.deltas[i],self.activations[i-1],self.weights[i])


        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.activations[len(self.layers)-1][k])**2
        return error
        
    def calcHDelta(self, activation1=[], delta1=[],delta2=[], weightsA1A2 =[[]]):
        for j in range(len(delta1)):
            error = 0.0
            for k in range(len(delta2)):
                error = error + delta2[k]*weightsA1A2[j][k]
            delta1[j] = dsigmoid(activation1[j]) * error
        return delta1
            
    def weightUpdate(self,delta2=[],activation1=[],weightA1A2 = [[]]):
        for i in range(len(delta2)):
            for j in range(len(activation1)):
                change = delta2[j]*activation1[i]
                weightA1A2[i][j] = weightA1A2[i][j] + self.LR*change
        return weightA1A2

    def test(self, patterns):
        counter = 0
        for p in patterns:
            inputs = p[0]
            targets = p[1]
            guess = self.feedforward(inputs)
            
            if guess[0] >= .5  and targets[0] == 1:
                counter+=1
            elif guess[0] < .5  and targets[0] == 0:
                counter+=1  
        print((counter/len(patterns)) * 100, "% classification accuracy")
    
    def test2(self, patterns):
        print("1 means outside circle, 0 inside")
        for p in patterns:
            print(p[0], '->', self.feedforward(p[0]))          
            
            

    def Save_Weights(self):
        file_ = open('weights.txt', 'w')
        file_.write(self.weightsHO, self.weightsH1H2,self.weightsHO)
        file_.close()
        

    def train(self, patterns, iterations=10):

        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedforward(inputs)
                error = error + self.backPropagate(targets,self.LR)
            print("epoch:", i, 'error: %-.5f' % error)

def demo():
    trainingset=[]
    radius = 2
    counter=0
    while counter < 100:
        point_x = rand(-5,5)
        point_y = rand(-5,5)
        if (point_x**2 + point_y**2) < radius**2:
            trainingset.append([[point_x,point_y],[0]])
        else:
            trainingset.append([[point_x,point_y],[1]])
        counter = counter + 1
   
    
    testset=[]
    counter=0
    while counter < 10:
        point_x = rand(-5,5)
        point_y = rand(-5,5)
        if (point_x**2 + point_y**2) < radius**2:
            testset.append([[point_x,point_y],[0]])
        else:
            testset.append([[point_x,point_y],[1]])
        counter = counter + 1
    
    size=int(input("Enter the number of layers: "))
    layers = [0.0] * size
    for i in range (0, size):
        layers[i] = input("enter the size of the ith layer")
        
    
    n = NeuralNet(2,layers,1,.5)
    
    n.test(testset)
   
    n.train(trainingset)
  
    #n.test2(testset)
    n.test(testset)
demo()
