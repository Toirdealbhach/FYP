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
    def __init__(self, numInputs, numHidden, numHidden2, numOutputs):
        # number of input, hidden, hidden2 and output nodes
        self.numInputs = numInputs + 1 # +1 for bias node
        self.numHidden = numHidden
        self.numHidden2 = numHidden2
        self.numOutputs = numOutputs

        # number of times activations are done
        self.inputActivation = [1.0]*self.numInputs
        self.hiddenActivation = [1.0]*self.numHidden
        self.hiddenActivation2 = [1.0]*self.numHidden2
        self.outputActivation = [1.0]*self.numOutputs
        
        # create weights
        self.weightsIH = makeMtx(self.numInputs, self.numHidden)
        self.weightsH1H2 = makeMtx(self.numHidden,self.numHidden2)
        self.weightsHO = makeMtx(self.numHidden2, self.numOutputs)
        # set them to random vaules
        for i in range(0,self.numInputs):
            for j in range(0,self.numHidden):
                self.weightsIH[i][j] = rand(-0.2, 0.2)
        
        for i in range(0,self.numHidden):
            for j in range(0,self.numHidden2):
                self.weightsH1H2[i][j] = rand(-0.2, 0.2)
                
        for j in range(0,self.numHidden2):
            for k in range(0,self.numOutputs):
                self.weightsHO[j][k] = rand(-0.2, 0.2)

          
        self.changeIH = makeMtx(self.numInputs, self.numHidden)
        self.changeH1H2 = makeMtx(self.numHidden, self.numHidden2)
        self.changeHO = makeMtx(self.numHidden2, self.numOutputs)

    def feedforward(self, inputs):
        if len(inputs) != self.numInputs-1:
            raise ValueError('The input nodes should be the same length as the data input length. E.G. a point classifying system needs input nodes = dimension of the points in the training set ')

        # input nodes
        for i in range(self.numInputs-1):
            self.inputActivation[i] = inputs[i]

        # hidden node activations
        for j in range(self.numHidden):
            sumOf = 0.0
            for i in range(self.numInputs):
                sumOf = sumOf + self.inputActivation[i] * self.weightsIH[i][j]
            self.hiddenActivation[j] = sigmoid(sumOf)
        
        # hidden2 node activations
        for j in range(self.numHidden2):
            sumOf = 0.0
            for i in range(self.numHidden):
                sumOf = sumOf + self.hiddenActivation[i] * self.weightsH1H2[i][j]
            self.hiddenActivation2[j] = sigmoid(sumOf)
        # output node activations
        for k in range(self.numOutputs):
            sumOf = 0.0
            for j in range(self.numHidden2):
                sumOf = sumOf + self.hiddenActivation2[j] * self.weightsHO[j][k]
            self.outputActivation[k] = sigmoid(sumOf)
        #return guess
        return self.outputActivation[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.numOutputs:
            raise ValueError('wrong number of target values')

        # calculate error for output
        output_deltas = [0.0] * self.numOutputs
        for k in range(self.numOutputs):
            error = targets[k]-self.outputActivation[k]
            output_deltas[k] = dsigmoid(self.outputActivation[k]) * error
        
        # calculate error for hidden layer 2
        hidden_deltas2 = [0.0] * self.numHidden2
        for j in range(self.numHidden2):
            error = 0.0
            for k in range(self.numOutputs):
                error = error + output_deltas[k]*self.weightsHO[j][k]
            hidden_deltas2[j] = dsigmoid(self.hiddenActivation2[j]) * error
        
        # calculate error for hidden layer 1
        hidden_deltas = [0.0] * self.numHidden
        for j in range(self.numHidden):
            error = 0.0
            for k in range(self.numHidden2):
                error = error + hidden_deltas2[k]*self.weightsH1H2[j][k]
            hidden_deltas[j] = dsigmoid(self.hiddenActivation[j]) * error

        # update weights between hidden layer 2 and output
        for j in range(self.numHidden2):
            for k in range(self.numOutputs):
                change = output_deltas[k]*self.hiddenActivation2[j]
                self.weightsHO[j][k] = self.weightsHO[j][k] + N*change + M*self.changeHO[j][k]
                self.changeHO[j][k] = change
                #print N*change, M*self.changeHO[j][k]
        
        # update weights between hidden layer 1 and hidden layer 2
        for j in range(self.numHidden):
            for k in range(self.numHidden2):
                change = hidden_deltas2[k]*self.hiddenActivation[j]
                self.weightsH1H2[j][k] = self.weightsH1H2[j][k] + N*change + M*self.changeH1H2[j][k]
                self.changeH1H2[j][k] = change
        
        # update weights between input and hidden layer 1
        for i in range(self.numInputs):
            for j in range(self.numHidden):
                change = hidden_deltas[j]*self.inputActivation[i]
                self.weightsIH[i][j] = self.weightsIH[i][j] + N*change + M*self.changeIH[i][j]
                self.changeIH[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.outputActivation[k])**2
        return error


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
        

    def train(self, patterns, iterations=100, N=0.5, M=0.1):

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
    while counter < 1000:
        point_x = rand(-5,5)
        point_y = rand(-5,5)
        if (point_x**2 + point_y**2) < radius**2:
            trainingset.append([[point_x,point_y],[0]])
        else:
            trainingset.append([[point_x,point_y],[1]])
        counter = counter + 1
   
    #test it with a randomly generated set of points inside or outside a cirlce
    testset=[]
    counter=0
    while counter < 120:
        point_x = rand(-5,5)
        point_y = rand(-5,5)
        if (point_x**2 + point_y**2) < radius**2:
            testset.append([[point_x,point_y],[0]])
        else:
            testset.append([[point_x,point_y],[1]])
        counter = counter + 1
    
    #Create a neural network with 2 hidden layers
    n = NeuralNet(2, 50, 50, 1)
    #test it with a randomly generated set of points inside or outside a cirlce
    #n.test2(testset)
    n.test(testset)
    # train it with a randomly generated set of points inside or outside a cirlce
    n.train(trainingset)
    # test it
    n.test2(testset)
    n.test(testset)
demo()
