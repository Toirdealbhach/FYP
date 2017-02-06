
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <math.h>

using namespace std;

//======================network architecture=====================

const int numInputs = 3;         // Input nodes, plus the bias input.
const int numHidden = 100;       //number of nodes in hidden layer 1
const int numHidden2 = 100;		//number of nodes in hidden layer 2
const int numOutput = 1;		//number of output nodes

//========================device copies of network architecture===========
__device__ __constant__ int dev_numInputs = 3;
__device__ __constant__ int dev_numHidden = 100;
__device__ __constant__ int dev_numHidden2 = 100;
__device__ __constant__ int dev_numOutput = 1;
//=========================device copies network parameters=====================                             
__device__ __constant__ double LR = 0.5;       // Learning rate                   
__device__ __constant__ double M = 0.1;      // Momentum Rate

//=========================dataset variables======================
const int numPatterns = 100; // number of input patterns for circle experiment.
const int radius = 1;          //radius of the circle
const int minRange = -2;       //input range for points in the dataset
const int maxRange = 2;
const int numEpochs = 1000;    //Amount of training to do. epoch = 1 exposure to the entire training set


__device__ double dev_Guess = 0.0;             // network output value.
__device__ double dev_errThisPat = 0.0;
__device__ double dev_RMSerror = 0.0;      // Squared error
__device__ double dev_errorTotal = 0.0;

 int patNum = 0;
 int *dev_patNum; //tracking the pattern number
 double Guess = 0.0;             // network output value.
 double errThisPat = 0.0;
 double RMSerror = 0.0;      // Squared error
 double errorTotal = 0.0;
 double SumOf = 0;

 double **trainInputs;
 double *trainOutput = new double[numPatterns];           // "Actual" output values.




//==================Matrices Vital for Backprop================
//may need to malloc
double *hiddenActivation = new double[numHidden];
double *hidden2Activation= new double[numHidden2];
double *outputActivation = new double[numOutput];

double **weightsIH = (double**)malloc((numHidden * numInputs) * sizeof(double)); // Input to Hidden weights.
double **weightsH1H2 = (double**)malloc((numHidden * numHidden2) * sizeof(double));
double **weightsHO = (double**)malloc((numOutput * numHidden2) * sizeof(double)); // Hidden to Output weights.

double *outputDeltas=new double[numOutput];
double *hidden2Deltas=new double[numHidden2];
double *hiddenDeltas=new double[numHidden];


double **changeHidden = (double**)malloc((numHidden * numInputs) *sizeof(double));
double **changeHidden2 = (double**)malloc((numHidden * numHidden2) *sizeof(double));
double **changeOutput = (double**)malloc((numOutput * numHidden2) *sizeof(double));

//=================device versions of vectors=================
double *dev_hiddenActivation = 0;
double *dev_hidden2Activation = 0;
double *dev_outputActivation = 0;

double **dev_weightsIH = 0;
double **dev_weightsH1H2 = 0;
double **dev_weightsHO = 0;

double *dev_outputDeltas = 0;
double *dev_hiddenDeltas = 0;
double *dev_hidden2Deltas = 0;

double **dev_changeHidden = 0;
double **dev_changeHidden2 = 0;
double **dev_changeOutput = 0;

double **dev_trainInputs = 0;
double *dev_trainOutput = 0;
//========================pitch sizes for 2D vector memory allocation=============================
size_t pitch_trainInputs;
size_t pitch_WeightsIH;
size_t pitch_WeightsH1H2;
size_t pitch_WeightsHO;
size_t pitch_changeHidden;
size_t pitch_changeHidden2;
size_t pitch_changeOutput;


//===================================Function Prototypes===================

void initWeights();
void feedforward();
void backProp();
cudaError_t initData();
//void test_1();
double getRand(double num1, double num2);
double sigmoid(double x);
double dsigmoid(double x);
cudaError_t DeviceMemoryPrep(int numInputs,
	double hiddenActivation[], int numHidden,
	double hidden2Activation[], int numHidden2,
	double outputActivation[], int numOutput,
	double **weightsIH, double **weightsH1H2, double **weightsHO,
	double outputDeltas[], double hiddenDeltas[], double hidden2Deltas[],
	double **changeHidden, double **changeHidden2, double **changeOutput);
cudaError_t train();
__global__ void CudaBackProp_part1(double dev_outputDeltas[], double dev_outputActivation[]);
__global__ void CudaBackProp_part2(double dev_outputDeltas[], double**dev_weightsHO, double dev_hidden2Deltas[], double dev_hidden2Activation[]);
__global__ void CudaBackProp_part3(double dev_Hidden2Deltas[], double**dev_weightsH1H2, double dev_hiddenDeltas[], double hiddenActivation[]);
__global__ void CudaBackProp_part4(double dev_outputDeltas[], double dev_hidden2Activation[], double**dev_weightsHO, double**dev_changeOutput);
__global__ void CudaBackProp_part5(double dev_hidden2Deltas[], double dev_hiddenActivation[], double**dev_weightsH1H2, double**dev_changeHidden2);
__global__ void CudaBackProp_part6(int *patnum, double dev_hiddenDeltas[], double** dev_trainInputs, double**dev_changeHidden, double**dev_weightsIH);
__global__ void CudaFeedForward_part1(int *patnum, double**dev_trainInputs, double**dev_weightsIH, double dev_hiddenActivation[]);
__global__ void CudaFeedForward_part2( double dev_hiddenActivation[], double**dev_weightsH1H2, double dev_hidden2Activation[]);
__global__ void CudaFeedForward_part3(int *patnum, double dev_hidden2Activation[], double**dev_weightsHO, double dev_outputActivation[], double dev_trainOutput[]);


int main()
{   
	cudaError_t cudaStatus;
	//srand((unsigned)time(0));   // Seed the random num generator
	initWeights();
	cout << "created weights" << endl;

	initData();
	cout << "created Training data" << endl;

	cudaStatus = train();
	if (cudaStatus != cudaSuccess) {
		cout << stderr << " Training Function Failed" << endl;
		return 1;
	}
	cout << "Finished training" << endl;

	//Training has finished.
	cout << "testing" << endl;
	//test_1();
	
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	//----------------------------------------------------------------------
	free(weightsIH);
	free(weightsH1H2);
	free(weightsHO);
	free(changeHidden);
	free(changeOutput);
	free(changeHidden2);



    return 0;
}

/* Helper function for Memory Allocation.
INPUT:Takes all backpropagation arrays as inputs, aswell as variables describing dimensions.
OUTPUT: outputs a cudaError_t type, describing any errors that may have happened during operation.
FUNCTIONALITY: allocates memory on the device for all relevant arrays for backprop and feedforward functions.
*/
cudaError_t DeviceMemoryPrep(int numInputs,
							 double hiddenActivation[], int numHidden, 
							 double hidden2Activation[], int numHidden2,
							 double outputActivation[], int numOutput,
							 double **weightsIH, double **weightsH1H2, double **weightsHO, 
							 double outputDeltas[], double hiddenDeltas[], double hidden2Deltas[],
							 double **changeHidden, double **changeHidden2, double **changeOutput
							)
{
	cudaError_t cudaStatus;

    // =====================================Choose which GPU to run on============================================
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //=========================================Allocate 1D GPU buffers for vectors==========================================
    cudaStatus = cudaMalloc((void**)&dev_hiddenActivation, numHidden * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc1 failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_hidden2Activation, numHidden2 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc2 failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_outputActivation, numOutput * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc3 failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_outputDeltas, numOutput * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc4 failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_hiddenDeltas, numHidden * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc5 failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_hidden2Deltas, numHidden2 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc6 failed!");
		goto Error;
	}

	//========================================Allocate 2D GPU buffers for 2D vectors==========================================
	//T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column; <---- addressing a "ptich"
	
	cudaStatus = cudaMallocPitch((void **)&dev_weightsIH,
		&pitch_WeightsIH,
		numInputs * sizeof(double),
		numHidden * sizeof(double)
	);
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMallocPitch1 failed!");
			goto Error;
		}

	
	cudaStatus = cudaMallocPitch((void **)&dev_weightsH1H2,
		&pitch_WeightsH1H2,
		numHidden * sizeof(double),
		numHidden2 * sizeof(double)
	);
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMallocPitch2 failed!");
			goto Error;
		}

	
	cudaStatus = cudaMallocPitch((void **)&dev_weightsHO,
		&pitch_WeightsHO,
		numHidden2 * sizeof(double),
		numOutput * sizeof(double)
	);
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMallocPitch3 failed!");
			goto Error;
		}

	
	cudaStatus = cudaMallocPitch((void **)&dev_changeHidden,
		&pitch_changeHidden,
		numInputs * sizeof(double),
		numHidden * sizeof(double)
	);
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMallocPitch4 failed!");
			goto Error;
		}

	
	cudaStatus = cudaMallocPitch((void **)&dev_changeHidden2,
		&pitch_changeHidden2,
		numHidden * sizeof(double),
		numHidden2 * sizeof(double)
	);
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMallocPitch5 failed!");
			goto Error;
		}

	
	cudaStatus = cudaMallocPitch((void **)&dev_changeOutput,
		&pitch_changeOutput,
		numHidden2 * sizeof(double),
		numOutput * sizeof(double)
	);
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMallocPitch6 failed!");
			goto Error;
		}

	//-------------------------------------------------------------------------------------

    //=======================================Copy 1D vectors from host memory to GPU buffers======================================
    cudaStatus = cudaMemcpy(dev_hiddenActivation, hiddenActivation, numHidden * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy1 failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_hidden2Activation, hidden2Activation, numHidden2 * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy2 failed!");
        goto Error;
    }
	cudaStatus = cudaMemcpy(dev_outputActivation, outputActivation, numOutput * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3 failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_outputDeltas, outputDeltas, numOutput * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy4 failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_hiddenDeltas, hiddenDeltas, numHidden * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy5 failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_hidden2Deltas, hidden2Deltas, numHidden2 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy6 failed!");
		goto Error;
	}

	//===========================================Copy 2D vectors=============================================
	cudaStatus = cudaMemcpy2D(dev_weightsIH, pitch_WeightsIH*sizeof(double), weightsIH, pitch_WeightsIH,
		numInputs * sizeof(double), numHidden, cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D 1 failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy2D(dev_weightsH1H2, pitch_WeightsH1H2 * sizeof(double), weightsH1H2, pitch_WeightsH1H2,
		numHidden * sizeof(double),numHidden2 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D 2 failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy2D(dev_weightsHO, pitch_WeightsHO * sizeof(double), weightsHO, pitch_WeightsHO,
		numHidden2 * sizeof(double),numOutput * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D 3 failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy2D(dev_changeHidden, pitch_changeHidden * sizeof(double), changeHidden, pitch_changeHidden,
		numInputs * sizeof(double),numHidden * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D 4 failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy2D(dev_changeHidden2, pitch_changeHidden2 * sizeof(double), changeHidden2,pitch_changeHidden2,
		numHidden * sizeof(double),numHidden2 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D 5 failed!");
		goto Error;
	}
	
	cudaStatus = cudaMemcpy2D(dev_changeOutput, pitch_changeOutput * sizeof(double), changeOutput, pitch_changeOutput,
		numHidden2 * sizeof(double),numOutput * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D 6 failed!");
		goto Error;
	}

Error:
	cudaFree(dev_hiddenActivation);
	cudaFree(dev_hidden2Activation);
	cudaFree(dev_outputActivation);

	cudaFree(dev_weightsIH);
	cudaFree(dev_weightsH1H2);
	cudaFree(dev_weightsHO);

	cudaFree(dev_outputDeltas);
	cudaFree(dev_hiddenDeltas);
	cudaFree(dev_hidden2Deltas);

	cudaFree(dev_changeHidden);
	cudaFree(dev_changeHidden2);
	cudaFree(dev_changeOutput);



	//----------------------------------------------------------------------------------------------------------------
	return cudaStatus;
    
}

cudaError_t train()
{
	cudaError_t cudaStatus = DeviceMemoryPrep(numInputs,
		hiddenActivation, numHidden,
		hidden2Activation, numHidden2,
		outputActivation, numOutput,
		weightsIH, weightsH1H2, weightsHO,
		outputDeltas, hiddenDeltas, hidden2Deltas,
		changeHidden, changeHidden2, changeOutput);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "DeviceMemoryPrep failed!");
		return cudaStatus;
	}
	
	for (int j = 0; j <= numEpochs; j++)
	{
		errorTotal = 0;
		for (int i = 0; i < numPatterns; i++)
		{
			patNum = i;
			cudaMalloc((void**)&dev_patNum, sizeof(int));
			cudaMemcpy(dev_patNum, &patNum, sizeof(int), cudaMemcpyHostToDevice);
			//Calculate the output and error for this pattern.
			feedforward();
			backProp();
			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
		}
		errorTotal = sqrt(errorTotal / numPatterns);
		if (j % 100 == 0) { cout << "epoch = " << j << " Error = " << errorTotal << endl; }
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	//-------------------------Copy output weight vectors from GPU buffer to host memory.-------------------------

	cudaStatus = cudaMemcpy2D(weightsIH, numHidden * sizeof(double), dev_weightsIH, pitch_WeightsIH,
		numInputs * sizeof(double), numHidden, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D return failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy2D(weightsH1H2, numHidden * sizeof(double), dev_weightsH1H2, pitch_WeightsH1H2,
		numHidden * sizeof(double), numHidden2 * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D return failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy2D(weightsHO, numHidden2 * sizeof(double), dev_weightsHO, pitch_WeightsHO,
		numHidden2 * sizeof(double), numOutput * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D return failed!");
		goto Error;
	}

	//-----------------------------------------------------------------------------------------
Error:
	cudaFree(dev_hiddenActivation);
	cudaFree(dev_hidden2Activation);
	cudaFree(dev_outputActivation);

	cudaFree(dev_weightsIH);
	cudaFree(dev_weightsH1H2);
	cudaFree(dev_weightsHO);

	cudaFree(dev_outputDeltas);
	cudaFree(dev_hiddenDeltas);
	cudaFree(dev_hidden2Deltas);

	cudaFree(dev_changeHidden);
	cudaFree(dev_changeHidden2);
	cudaFree(dev_changeOutput);
	return cudaStatus;
	
}

void backProp()
{
	
	CudaBackProp_part1<<<1, numOutput >>>(dev_outputDeltas, dev_outputActivation);
	
	CudaBackProp_part2<<<1, numHidden2 >>>(dev_outputDeltas, dev_weightsHO, dev_hidden2Deltas, dev_hidden2Activation);
	
	CudaBackProp_part3<<<1, numHidden >>>(dev_hidden2Deltas, dev_weightsH1H2, dev_hiddenDeltas,  hiddenActivation);
	
	CudaBackProp_part4<<<1, numHidden2 >>>(dev_outputDeltas, dev_hidden2Activation, dev_weightsHO, dev_changeOutput);
	
	CudaBackProp_part5<<<1, numHidden >>>(dev_hidden2Deltas, dev_hiddenActivation, dev_weightsH1H2, dev_changeHidden2);
	
	CudaBackProp_part6<<<1, numInputs >>>(dev_patNum, dev_hiddenDeltas,  dev_trainInputs ,dev_changeHidden, dev_weightsIH);
	
}

void initWeights()
{
	//=====================================initialise 2d arrays=========================================
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
		weightsH1H2[i] = new double[numOutput];
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
			weightsIH[i][j] = getRand(-.2, .2);
		}
		weightsIH[numInputs - 1][j] = 1;//Bias
	}
	for (int j = 0; j < numHidden2; j++)
	{
		for (int i = 0; i < numHidden; i++)
		{
			weightsH1H2[i][j] = getRand(-0.4, 0.4);
		}
	}
	for (int j = 0; j < numOutput; j++)
	{
		for (int i = 0; i < numHidden2; i++)
		{
			weightsHO[i][j] = getRand(-0.4, 0.4);
		}
	}
}

cudaError_t initData()
{
	cudaError_t cudaStatus;
	trainInputs = new double*[numPatterns];
	for (int i = 0; i < numPatterns; i++)
	{
		trainInputs[i] = new double[numInputs];
	}
	//data set of points inside/outside a circle
	for (int j = 0; j < numPatterns; j++){
		for (int i = 0; i < numInputs; i++){
			trainInputs[i][j] = getRand(minRange, maxRange);
		}

		if (((trainInputs[j][0] * trainInputs[j][0]) + (trainInputs[j][1] * trainInputs[j][1])) < (radius*radius)){//if in circle answer is 1
			trainOutput[j] = 1;
		}//otherwise answer is 0
		else { 
			trainOutput[j] = 0; 
		}
	}

	//==============================allocate and copy memory for output vector===========================
	cudaStatus = cudaMalloc((void**)&dev_trainOutput, numPatterns * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc output vec failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_trainOutput, dev_trainOutput, numPatterns * sizeof(double), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy output vec failed!");
		goto Error;
	}

	//===========================allocate and copy memory for input matrix============================
	cudaStatus = cudaMallocPitch((void **)&dev_trainInputs,
		&pitch_trainInputs,
		numPatterns * sizeof(double),
		numInputs * sizeof(double)
	);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMallocPitch1 failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy2D(dev_trainInputs, pitch_trainInputs * sizeof(double), trainInputs, pitch_trainInputs,
		numPatterns * sizeof(double), numInputs, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D 1 failed!");
		goto Error;
	}
	
	
Error:
	cudaFree(dev_trainInputs);
	cudaFree(dev_trainOutput);

	return cudaStatus;

}

void feedforward()
{
	//input nodes dont have activations, except an assignment from raw to an array
	//hidden nodes activations
	CudaFeedForward_part1<<<1, numHidden >>>(dev_patNum, dev_trainInputs, dev_weightsIH,  dev_hiddenActivation);
	
	CudaFeedForward_part2<<<1, numHidden2 >>>(dev_hiddenActivation, dev_weightsH1H2,  dev_hidden2Activation);
	
	CudaFeedForward_part3<<<1, numOutput >>>(dev_patNum,dev_hidden2Activation, dev_weightsHO, dev_outputActivation, dev_trainOutput);
	
	//Author Note:Needs general case for more than 1 output


}
/*
void test_1()
{
	for (int i = 0; i < numPatterns / 10; i++)
	{
		patNum = i;
		feedforward();
		cout << "pattern = " << patNum + 1 <<
			"| actual answer = " << trainOutput[patNum] <<
			"| Neural Net guess = " << Guess << endl;
	}
}*/

double getRand(double num1, double num2)
{
	return double((num1 - num2)*(rand() / double(RAND_MAX)) + num1); //(b-a)*random.random() + a
}

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

__device__ double dev_sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

__device__ double dev_dsigmoid(double x)
{
	return dev_sigmoid(x) * (1 - dev_sigmoid(x));
}

/*void test_2()
{
	int counter = 0;
	for (int i = 0; i < numPatterns / 10; i++)
	{
		patNum = i;
		feedforward();


		if (Guess >= .5 && trainOutput[patNum] == 1)
		{
			counter += 1;
		}
		else if (Guess < .5 && trainOutput[patNum] == -1)
		{
			counter += 1;
		}
	}
	cout << ((counter / (numPatterns / 10)) * 100) << "% classification accuracy" << endl;
}*/

//==================================Kernel Definitions=============================
__global__ void CudaBackProp_part1(double dev_outputDeltas[], double dev_outputActivation[])
{
    int i = threadIdx.x;
	dev_outputDeltas[i] = LR * dev_errThisPat * dev_outputActivation[i];
}

__global__ void CudaBackProp_part2(double dev_outputDeltas[], double**dev_weightsHO,double dev_hidden2Deltas[],double dev_hidden2Activation[])
{
	int j = threadIdx.x;
	double errors = 0;
	for (int k = 0; k < numOutput; k++)
	{
		errors = errors + dev_outputDeltas[k] * dev_weightsHO[j][k];
	}
	dev_hidden2Deltas[j] = dev_dsigmoid(dev_hidden2Activation[j]) * errors;
}

__global__ void CudaBackProp_part3(double dev_hidden2Deltas[], double**dev_weightsH1H2,double dev_hiddenDeltas[],double hiddenActivation[])
{
	int j = threadIdx.x;
	double errors = 0;
	for (int k = 0; k < dev_numHidden2; k++)
	{
		errors = errors + dev_hidden2Deltas[k] * dev_weightsH1H2[j][k];
	}
	dev_hiddenDeltas[j] = dev_dsigmoid(hiddenActivation[j]) * errors;
}

__global__ void CudaBackProp_part4(double dev_outputDeltas[], double dev_hidden2Activation[], double**dev_weightsHO, double**dev_changeOutput)
{
	int j = threadIdx.x;
	double change = 0;
	for (int k = 0; k< numOutput; k++)
	{
		change = dev_outputDeltas[k] * dev_hidden2Activation[j];
		dev_weightsHO[j][k] = dev_weightsHO[j][k] + LR*change + M*dev_changeOutput[j][k];
		dev_changeOutput[j][k] = change;
	}
}

__global__ void CudaBackProp_part5(double dev_hidden2Deltas[], double dev_hiddenActivation[], double**dev_weightsH1H2,double**dev_changeHidden2)
{
	int j = threadIdx.x;
	double change = 0;
	for (int k = 0; k< numHidden2; k++)
	{
		change = dev_hidden2Deltas[k] * dev_hiddenActivation[j];
		dev_weightsH1H2[j][k] = dev_weightsH1H2[j][k] + LR*change + M*dev_changeHidden2[j][k];
		dev_changeHidden2[j][k] = change;
	}
}

__global__ void CudaBackProp_part6(int *patnum, double dev_hiddenDeltas[], double** dev_trainInputs, double**dev_changeHidden, double**dev_weightsIH)
{
	int j = threadIdx.x;
	double change = 0;
	for (int k = 0; k< numHidden; k++)
	{
		change = dev_hiddenDeltas[k] * dev_trainInputs[*patnum][j];
		dev_changeHidden[j][k] = change;
		dev_weightsIH[j][k] = dev_weightsIH[j][k] + LR*change + M*dev_changeHidden[j][k];
	}
}

__global__ void CudaFeedForward_part1(int *patnum, double**dev_trainInputs, double**dev_weightsIH, double dev_hiddenActivation[])
{
	int i = threadIdx.x;
	double SumOf = 0.0;
	for (int j = 0; j < dev_numInputs; j++)
	{
		SumOf = SumOf + (dev_trainInputs[*patnum][j] * dev_weightsIH[j][i]);
	}
	dev_hiddenActivation[i] = dev_sigmoid(SumOf);
}

__global__ void CudaFeedForward_part2(double dev_hiddenActivation[], double**dev_weightsH1H2, double dev_hidden2Activation[])
{
	int i = threadIdx.x;
	double SumOf = 0.0;
	for (int j = 0; j < dev_numHidden; j++)
	{
		SumOf = SumOf + (dev_hiddenActivation[j] * dev_weightsH1H2[j][i]);
	}
	dev_hidden2Activation[i] = dev_sigmoid(SumOf);
}

__global__ void CudaFeedForward_part3(int *patnum, double dev_hidden2Activation[], double**dev_weightsHO, double dev_outputActivation[],double dev_trainOutput[])
{
	int i = threadIdx.x;
	double SumOf = 0.0;
	for (int j = 0; j < dev_numHidden2; j++)
	{
		SumOf = SumOf + (dev_hidden2Activation[j] * dev_weightsHO[j][i]);
	}
	dev_outputActivation[i] = dev_sigmoid(SumOf);
	dev_errThisPat = dev_trainOutput[*patnum] - dev_outputActivation[0];
	dev_RMSerror += dev_errThisPat * dev_errThisPat;
}


