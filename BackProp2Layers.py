
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h> 

using namespace std;
//==============================Function Prototypes================================
double getRand();
double** initData(double **trainInput, double trainOutput[], int numPatterns, int numInputs, int numOutput);

class NeuralNet {
	int numInputs;
	int numHidden;
	int numHidden2;
	int numOutput;
	int patNum = 0;

	const double LR_IH = 0.7;      
	const double LR_HO = 0.07;
	
	double errThisPat = 0.0;
	double outPred = 0.0;                   // "Expected" output values.
	double RMSerror = 0.0;// Root Mean Squared error.
	
	//==================Matrices Vital for Backprop================
	double *hiddenActivation; 
	double *hidden2Activation; 
	double *outputActivation; 

	double **weightsIH; 
	double **weightsH1H2; 
	double **weightsHO; 

	double *outputDeltas; 
	double *hidden2Deltas; 
	double *hiddenDeltas; 


	double **changeHidden; 
	double **changeHidden2; 
	double **changeOutput;

	double **trainInputs;
	double *trainOutput; 
	
public:
	NeuralNet(int numInputs, int numHidden, int numHidden2, int numOutput);
	void initWeights();
	void calcNet(double *trainInputs, double *trainOutput);
	void WeightChangesHO();
	void WeightChangesHidden(int patNum, double *trainInputs);
	void calcOverallError(double **trainInputs, int numPatterns, double *trainOutput);
	void displayResults(double **trainInputs, int numPatterns, double *trainOutput);
	void train(int numEpochs, int numPatterns, double *trainInputs[], double *trainOutput);
	

};

NeuralNet::NeuralNet(int inumInputs,int inumHidden,int inumHidden2,int inumOutput){

	numInputs = inumInputs;
	numHidden = inumHidden;
	numHidden2 = inumHidden2;
	numOutput = inumOutput;


	hiddenActivation = new double[numHidden];
	hidden2Activation = new double[numHidden2];
	outputActivation = new double[numOutput];

	weightsIH = (double**)malloc((numHidden * numInputs) * sizeof(double)); // Input to Hidden weights.
	weightsH1H2 = (double**)malloc((numHidden * numHidden2) * sizeof(double));
	weightsHO = (double**)malloc((numOutput * numHidden2) * sizeof(double)); // Hidden to Output weights.

	outputDeltas = new double[numOutput];
	hidden2Deltas = new double[numHidden2];
	hiddenDeltas = new double[numHidden];


	changeHidden = (double**)malloc((numHidden * numInputs) * sizeof(double));
	changeHidden2 = (double**)malloc((numHidden * numHidden2) * sizeof(double));
	changeOutput = (double**)malloc((numOutput * numHidden2) * sizeof(double));
}
int main() {
	const int numPatterns = 4;      // Input patterns for XOR experiment.
	const int numEpochs = 200;
	int numInputs = 3;
	cout << "enter the size of the input layer" << endl;
	//cin >> numInputs;
	int numHidden=100;
	cout << "enter the size of hidden layer 1" << endl;
	//cin >> numHidden;
	int numHidden2=100;
	cout << "enter the size of hidden layer 2" << endl;
	//cin >> numHidden2;
	int numOutput=1;
	cout << "enter the size of the output layer" << endl;
	//cin >> numOutput;

	double **trainInputs = (double**)(malloc(numInputs * numPatterns * sizeof(double)));
	double *trainOutput = new double[numPatterns];

	srand((unsigned)time(0));   // Seed the generator with system time.

	NeuralNet NN(numInputs, numHidden, numHidden2, numOutput);
	NN.initWeights();

	trainInputs = initData(trainInputs, trainOutput, numPatterns ,numInputs,numOutput);

	NN.train(numEpochs,numPatterns, trainInputs, trainOutput );
	//Training has finished.
	NN.displayResults(trainInputs,numPatterns, trainOutput);

	return 0;

free:
	NN;
	trainInputs;
	trainOutput;
}

void NeuralNet::initWeights() {

	weightsIH = new double*[numInputs];
	for (int i = 0; i < numInputs; i++)
	{
		weightsIH[i] = new double[numHidden];
	}

	weightsH1H2 = new double*[numHidden];
	for (int i = 0; i < numHidden; i++)
	{
		weightsH1H2[i] = new double[numHidden2];
	}

	weightsHO = new double*[numHidden2];
	for (int i = 0; i < numHidden2; i++)
	{
		weightsHO[i] = new double[numOutput];
	}

	changeHidden = new double*[numInputs];
	for (int i = 0; i < numInputs; i++)
	{
		changeHidden[i] = new double[numHidden];
	}

	changeHidden2 = new double*[numHidden];
	for (int i = 0; i < numHidden; i++)
	{
		changeHidden2[i] = new double[numHidden2];
	}

	changeOutput = new double*[numHidden2];
	for (int i = 0; i < numHidden2; i++)
	{
		changeOutput[i] = new double[numOutput];
	}
	//=================================Initialize weights to random values.======================================
	for (int j = 0; j < numHidden; j++)
	{
		for (int i = 0; i < numInputs - 1; i++)
		{
			weightsIH[i][j] = (getRand() - 0.5) / 5;
		}
		weightsIH[numInputs - 1][j] = 1;//Bias
	}
	for (int j = 0; j < numHidden2; j++)
	{
		for (int i = 0; i < numHidden; i++)
		{
			weightsH1H2[i][j] = (getRand() - 0.5) / 5;;
		}
	}
	for (int j = 0; j < numOutput; j++)
	{
		for (int i = 0; i < numHidden2; i++)
		{
			weightsHO[i][j] = (getRand() - 0.5) / 2;
		}
	}
}
void NeuralNet::train(int numEpochs, int numPatterns,double *trainInputs[], double *trainOutput) {
	// Train the network.
	for (int j = 0; j <= numEpochs; j++) {

		for (int i = 0; i < numPatterns; i++) {

			//Select a pattern at random.
			patNum = rand() % numPatterns;

			//Calculate the output and error for this pattern.
			double *temp = trainInputs[patNum];
			calcNet(trainInputs[patNum],trainOutput);

			//Adjust network weights.
			WeightChangesHO();
			WeightChangesHidden(patNum, trainInputs[patNum]);
		}

		calcOverallError(trainInputs, numPatterns, trainOutput);

		//Display the overall network error after each epoch
		cout << "epoch = " << j << " RMS Error = " << RMSerror << endl;
	}
}

double** initData(double **trainInput, double trainOutput[], int numPatterns, int numInputs, int numOutput) {
	// The data here is the XOR data which has been rescaled to 
	// the range -1 to 1.

	// An extra input value of 1 is also added to act as the bias.

	// The output must lie in the range -1 to 1.
	double **trainInputs = new double*[numPatterns];
	for (int i = 0; i < numPatterns; i++)
	{
		trainInputs[i] = new double[numInputs];
	}
	trainInputs[0][0] =1;
	trainInputs[0][1] = -1;
	trainInputs[0][2] = 1; // Bias
	trainOutput[0] = 1;

	trainInputs[1][0] = -1;
	trainInputs[1][1] = 1;
	trainInputs[1][2] = 1; // Bias
	trainOutput[1] = 1;

	trainInputs[2][0] = 1;
	trainInputs[2][1] = 1;
	trainInputs[2][2] = 1; // Bias
	trainOutput[2] = -1;

	trainInputs[3][0] = -1;
	trainInputs[3][1] = -1;
	trainInputs[3][2] = 1; // Bias
	trainOutput[3] = -1;
	return trainInputs;
}

void NeuralNet::calcNet(double *trainInputs, double *trainOutput) {
	// Calculates values for Hidden and Output nodes.

	for (int i = 0; i < numHidden; i++) {
		hiddenActivation[i] = 0.0;
		for (int j = 0; j < numInputs; j++) {
			hiddenActivation[i] = hiddenActivation[i] + trainInputs[j] * weightsIH[j][i];
		}
		hiddenActivation[i] = tanh(hiddenActivation[i]);
	}

	for (int i = 0; i < numHidden2; i++) {
		hidden2Activation[i] = 0.0;
		for (int j = 0; j < numHidden; j++) {
			hidden2Activation[i] = hidden2Activation[i] + (hiddenActivation[j] * weightsH1H2[i][j]);
		}
		hiddenActivation[i] = tanh(hiddenActivation[i]);
	}

	outPred = 0.0;
	for (int i = 0; i < numHidden2; i++) {
		for (int j = 0; j < numOutput; j++) {
			outPred = outPred + hidden2Activation[j] * weightsHO[j][i];
		}
		outputActivation[i] = tanh(outPred);
	}
	//Calculate the error: "Expected" - "Actual"
	errThisPat = outPred - trainOutput[0];
}

void NeuralNet::WeightChangesHO() {
	//Adjust the Hidden to Output weights.
	for (int j = 0; j < numOutput; j++) {
		outputDeltas[j] = LR_HO * errThisPat * outputActivation[j];
		for (int k = 0; k < numHidden2; k++) {
			weightsHO[k][j] = weightsHO[k][j] - outputDeltas[j];
			// Regularization of the output weights.
			if (weightsHO[k][j] < -5) {
				weightsHO[k][j] = -5;
			}
			else if (weightsHO[k][j] > 5) {
				weightsHO[k][j] = 5;
			}
		}
	}
}

void NeuralNet::WeightChangesHidden(int patNum, double *trainInputs) {
	// Adjust the Input to Hidden weights.

	for (int i = 0; i < numHidden2; i++) {

		for (int k = 0; k < numHidden; k++) {

			double x = 1 - (hidden2Activation[i] * hidden2Activation[i]);
			x = x * weightsH1H2[i][k] * outputDeltas[0] * LR_IH;
			x = x * hiddenActivation[k];
			hidden2Deltas[i] = x;
			weightsH1H2[k][i] = weightsH1H2[k][i] - hidden2Deltas[i];

			if (weightsH1H2[k][i] < -5) {
				weightsH1H2[k][i] = -5;
			}
			else if (weightsH1H2[k][i] > 5) {
				weightsH1H2[k][i] = 5;
			}
		}
	}
	double sumof = 0;
	for (int j = 0; j < numHidden2; j++) {
		sumof = sumof + hidden2Deltas[j];
	}

	for (int i = 0; i < numHidden; i++) {

		for (int k = 0; k < numInputs; k++) {
			double x = 1 - (hiddenActivation[i] * hiddenActivation[i]);
			x = x * weightsIH[k][i] * sumof;
			x = x * trainInputs[k] * LR_IH;
			hiddenDeltas[i] = x;
			weightsIH[k][i] = weightsIH[k][i] - hiddenDeltas[i];

			if (weightsIH[k][i] < -5) {
				weightsIH[k][i] = -5;
			}
			else if (weightsIH[k][i] > 5) {
				weightsIH[k][i] = 5;
			}
		}
	}


}

void NeuralNet::calcOverallError(double ** trainInputs, int numPatterns, double *trainOutput) {
	RMSerror = 0.0;

	for (int i = 0; i < numPatterns; i++) {
		patNum = i;
		calcNet(trainInputs[patNum], trainOutput);
		RMSerror = RMSerror + (errThisPat * errThisPat);
	}

	RMSerror = RMSerror / numPatterns;
	RMSerror = sqrt(RMSerror);
}

void NeuralNet::displayResults(double ** trainInputs,int numPatterns,double *trainOutput) {
	for (int i = 0; i < numPatterns; i++) {
		patNum = i;
		calcNet(trainInputs[patNum], trainOutput);
		cout << "pat = " << patNum + 1 <<
			" actual = " << trainOutput[patNum] <<
			" neural model = " << outPred << endl;
	}
}

double getRand() {
	return double(rand() / double(RAND_MAX));
}
