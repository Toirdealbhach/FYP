import numpy as np
import random
import matplotlib.pyplot as plt


class Perceptron():
    
    def __init__(self,inputlen):
        self.inputlen = inputlen 
        self.weights = np.random.rand(inputlen)
        self.learningRate = 0.10
    
    @staticmethod
    def activate(n):
        if n>0:
            return 1
        elif n<0:
            return -1
        elif n==0:
            return 0
    
            
    def feedforward(self, inputs):
        sumOf=0
        for i in range (0,len(inputs)):
            sumOf += inputs[i]*self.weights[i]
        return self.activate(sumOf)
        
    def train(self,inputs, desired):
        guess = self.feedforward(inputs)
    
        error = desired - guess

        for i in range (0, self.inputlen):
            self.weights[i] += self.learningRate * self.weights[i] * error/100
        #print(self.weights)

p = Perceptron(3)

class Trainer():
    def __init__(self,x,y,a):
        self.inputs = np.array([x,y,1])
        self.answer = a
       
def f(x):
  return 2*x+1 #a particular line




  

  #(640, 360)
training = np.empty([2000], dtype = Trainer)
for i in range (0, 2000):
    x = random.randint(-640/2,640/2)
    y = random.randint(-360/2,360/2)
    
    answer = 1
    if (y < f(x)):
        answer = -1
    training[i] = Trainer(x, y, answer)


    
print("with no training done this is how well the perceptron fits the data")   
fig = plt.figure(1)
ax = fig.gca()
ax.plot([-100,100],[-201,201])

for i in range(0,2000):
    guess = p.feedforward(training[i].inputs);
        
    if guess > 0:                      
        ax.plot(training[i].inputs[0], training[i].inputs[1], 'g+' )
    elif guess < 0:
        ax.plot(training[i].inputs[0], training[i].inputs[1], 'r.' )

plt.close('all')
print("These graphs show the perceptron slowly learning")
for i in range (0, 2000000):
    p.train(training[i%2000].inputs, training[i%2000].answer)
    plt.close('all')
    #if i%10000000 == 0:

figs = plt.figure(2)
axs = figs.gca()
axs.plot([-100,1,2,3,100],[-201,3,5,7,201])
for j in range(0,2000):
    guess = p.feedforward(training[j].inputs);
    if guess > 0:                      
        axs.plot(training[j].inputs[0], training[j].inputs[1], 'g+' )
    elif guess < 0:
        axs.plot(training[j].inputs[0], training[j].inputs[1], 'r.' )
plt.show()
    

          
    
            
            
            
            
#point = np.array([50, -12,1])



#'result = p.feedforward(point)

#print(result)
