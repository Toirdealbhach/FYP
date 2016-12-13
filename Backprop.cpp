#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <math.h>

using namespace std;

const int numInputs = 3;
const int numHidden = 10;
const int numHidden2 = 5;
const int numOutput = 1;        // Input nodes, plus the bias input.

const double LR = 0.5;       // Learning rate input to hidden
       //learning rate hidden to output
const double M = 0.1;      // Momentum Rate

const int numPatterns = 10000; // number of input patterns for circle experiment.
const int radius = 2;
const int minRange = -3;
const int maxRange = 3;      
const int numEpochs = 10000;    //Amount of training to do.

int patNum = 0;
double errThisPat = 0.0;
double Guess = 0.0;                   // network output value.
double RMSerror = 0.0;
double errorTotal = 0.0;                // Root Mean Squared error.. not really though, more like cumulative error

//----------------Vital for Backprop------------------
double hiddenActivation[numHidden] = {0.0};
double outputActivation[numOutput] = {0.0};         

double weightsIH[numInputs][numHidden]; // Input to Hidden weights.
double weightsHO[numHidden][numOutput]; // Hidden to Output weights.

double outputDeltas[numOutput];
double hiddenDeltas[numHidden];

double changeHidden[numInputs][numHidden] = {0.0};
double changeOutput[numHidden][numOutput] = {0.0};
//----------------------------------------------------

int trainInputs[numPatterns][numInputs];
int trainOutput[numPatterns];           // "Actual" output values.

//-------------------------------Function Prototypes-------------------
void train();
void initWeights();
void feedforward();
void backProp();
void Error();
void initData();
void test_1();
double getRand(double num1, double num2);
double sigmoid(double x);
double dsigmoid(double x);
//--------------------------------------------------------
int main()
{
    srand((unsigned)time(0));   // Seed the generator 
    initWeights();
    cout << "created weights"<< endl;
    
    initData();
    cout << "created data and starting training"<< endl;
    
    train();
    cout << "Finished training"<< endl;
    
    //Training has finished.
    cout << "testing";
    test_1();

    return 0;
}
void train()
{
    for(int j = 0; j <= numEpochs; j++)
    {
        for(int i = 0; i < numPatterns; i++)
        {
            patNum = i;
            //Calculate the output and error for this pattern.
            feedforward();
            //Adjust network weights.
            backProp();
            errorTotal = errorTotal + RMSerror;
        }
        if(j%100==0){ cout << "epoch = " << j << " Error = " << RMSerror << endl;}
    }
}


void initWeights()
{
// Initialize weights to random values.
    for(int j = 0; j < numHidden; j++)
    {
        for(int i = 0; i < numInputs-1; i++)
        {
            weightsIH[i][j] = getRand(-0.4,0.4);
            //cout << "Weight = " << weightsIH[i][j] << endl;
        }
        weightsIH[j][numInputs-1] = 1;//Bias
    }
    for(int j = 0; j < numOutput; j++)
    {
        for(int i = 0; i < numHidden; i++)
        {
            weightsHO[i][j] = getRand(-0.4,0.4);
            //cout << "Weight = " << weightsIH[i][j] << endl;
        }
    }  
}

void initData()
{
    //Training set of points inside/outside a circle
    for (int j = 0; j < numPatterns; j++)
    {
        for (int i = 0; i < numInputs; i++)
        {
            trainInputs[i][j] = getRand(minRange,maxRange);
        }
        
        if (((trainInputs[j][0]*trainInputs[j][0]) + (trainInputs[j][1]*trainInputs[j][1])) < (radius*radius))
        {//if in circle answer is 1
            trainOutput[j] = 1;
        }//otherwise answer is zero
        else{trainOutput[j] = 0;}
    }
}

void feedforward()
{
     //input nodes dont have activations, except an assignment from raw to an array
     
     //hidden nodes activations
    for(int i = 0; i < numHidden; i++)
    {
	  hiddenActivation[i] = 0.0;
        for(int j = 0; j < numInputs; j++)
        {
	        hiddenActivation[i] = hiddenActivation[i] + (trainInputs[patNum][j] * weightsIH[j][i]);
        }
        hiddenActivation[i] = sigmoid(hiddenActivation[i]);
    }

   Guess = 0.0;
    //output nodes activations
    for(int i = 0; i < numOutput; i++)
    {   
        outputActivation[i] = 0.0; 
        for (int j =0; j < numHidden; j++)
        {
            outputActivation[i] = outputActivation[i] + (hiddenActivation[j] * weightsHO[j][i]);
        }
        outputActivation[i] = sigmoid(outputActivation[i]);
    }
    
    //Needs general case for more than 1 output
    Guess = outputActivation[0];
    errThisPat = Guess - trainOutput[patNum];
}

void backProp()
{
    double errors = 0;
    double change = 0;
    //calc weight delta for output
    for(int k = 0; k < numOutput; k++)
    {
        outputDeltas[k] = LR * errThisPat * outputActivation[k];
        //weightsHO[k] = weightsHO[k] - weightChange;
    }

// Adjust the Input to Hidden weights.
    for(int j = 0; j < numHidden; j++)
    {
        errors = 0;
        for(int k = 0; k < numOutput; k++)
        {
            errors = errors + outputDeltas[k]*weightsHO[j][k];
            //weightsIH[k][i] = weightsIH[k][i] - weightChange;
        }
        hiddenDeltas[j] = dsigmoid(hiddenActivation[j]) * errors;
    }
    
    for (int j = 0;j< numHidden;j++)
    {
        for (int k=0;k< numOutput;k++)
        {
                change = outputDeltas[k]*hiddenActivation[j];
                weightsHO[j][k] = weightsHO[j][k] + LR*change + M*changeOutput[j][k];
                changeOutput[j][k] = change;
        }
    }
    
    for (int j = 0;j<numInputs;j++)
    {
        for (int k=0;k< numHidden;k++)
        {
                change = hiddenDeltas[k]*trainInputs[patNum][j];
                weightsIH[j][k] = weightsIH[j][k] + LR*change + M*changeHidden[j][k];
                changeHidden[j][k] = change;
        }
    }
    
    for (int k = 0; k < numOutput;k++)
    {
        RMSerror = RMSerror + 0.5 * ((trainOutput[k]-outputActivation[k])*(trainOutput[k]-outputActivation[k]));
    }
    
}

void test_1()
{
    for(int i = 0; i < numPatterns/10; i++)
    {
        patNum = i;
        feedforward();
        cout << "pattern = " << patNum + 1 << 
                " actual answer = " << trainOutput[patNum] << 
                " Neural Net guess = " << Guess << endl;
    }
}

double getRand(double num1, double num2)
{
    return double((num1-num2)*(rand()/ double(RAND_MAX)) + num1); //(b-a)*random.random() + a
}

double sigmoid(double x)
{
        return 1 / (1 + exp(-x));
}
double dsigmoid(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}
