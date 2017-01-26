

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
const int numHidden = 100;
const int numHidden2 = 100;
const int numOutput = 1;

//========================device copies of network architecture===========
__device__ __constant__ int numInputs = 3;
__device__ __constant__ int numHidden = 100;
__device__ __constant__ int numHidden2 = 100;
__device__ __constant__ int numOutput = 1;
//=========================device copies network parameters=====================                             
__device__ __constant__ double LR = 0.5;       // Learning rate                   
__device__ __constant__ double M = 0.1;      // Momentum Rate

//=========================dataset variables======================
const int numPatterns = 100; // number of input patterns for circle experiment.
const int radius = 1;          //radius of the circle
const int minRange = -2;       //input range for points in the dataset
const int maxRange = 2;
const int numEpochs = 1000;    //Amount of training to do. epoch = 1 exposure to the entire training set

int patNum = 0;                 //tracking the pattern number
double Guess = 0.0;             // network output value.
double errThisPat = 0.0;
double RMSerror = 0.0;      // Squared error
double errorTotal = 0.0;
double SumOf = 0;

//==================Matrices Vital for Backprop================
//===================================================NEED TO BE COPIED TO DEVICE IN BACKPROPPREP FUNCTION========================================================
double hiddenActivation[numHidden] = { 0.0 };
double hidden2Activation[numHidden2] = { 0.0 };
double outputActivation[numOutput] = { 0.0 };

double weightsIH[numInputs][numHidden]; // Input to Hidden weights.
double weightsH1H2[numHidden][numHidden2];
double weightsHO[numHidden2][numOutput]; // Hidden to Output weights.

double outputDeltas[numOutput];
double hidden2Deltas[numHidden2];
double hiddenDeltas[numHidden];


double changeHidden[numInputs][numHidden] = { 0.0 };
double changeHidden2[numHidden][numHidden2] = { 0.0 };
double changeOutput[numHidden2][numOutput] = { 0.0 };
//----------------------------------------------------

double trainInputs[numPatterns][numInputs];
int trainOutput[numPatterns];           // "Actual" output values.

//=============================Function Prototypes===================
void train();
void initWeights();
void feedforward();
void backProp();
void initData();
void test_1();
double getRand(double num1, double num2);
double sigmoid(double x);
double dsigmoid(double x);
cudaError_t CudaBackProp();

__global__ void CudaBackProp_part1()
{
    int i = threadIdx.x;
	outputDeltas[i] = LR * errThisPat * outputActivation[i];
}

__global__ void CudaBackProp_part2()
{
	int j = threadIdx.x;

	errors = 0;
	for (int k = 0; k < numOutput; k++)
	{
		errors = errors + outputDeltas[k] * weightsHO[j][k];
	}
	hidden2Deltas[j] = dsigmoid(hidden2Activation[j]) * errors;
}

__global__ void CudaBackProp_part3()
{
	int j = threadIdx.x;
	errors = 0;
	for (int k = 0; k < numHidden2; k++)
	{
		errors = errors + hidden2Deltas[k] * weightsH1H2[j][k];
	}
	hiddenDeltas[j] = dsigmoid(hiddenActivation[j]) * errors;
}

__global__ void CudaBackProp_part4()
{
	int j = threadIdx.x;
	for (int k = 0; k< numOutput; k++)
	{
		change = outputDeltas[k] * hidden2Activation[j];
		weightsHO[j][k] = weightsHO[j][k] + LR*change + M*changeOutput[j][k];
		changeOutput[j][k] = change;
	}
}

__global__ void CudaBackProp_part5()
{
	int j = threadIdx.x;
	for (int k = 0; k< numHidden2; k++)
	{
		change = hidden2Deltas[k] * hiddenActivation[j];
		weightsH1H2[j][k] = weightsH1H2[j][k] + LR*change + M*changeHidden2[j][k];
		changeHidden2[j][k] = change;
	}
}

__global__ void CudaBackProp_part6()
{
	int j = threadIdx.x;
	for (int k = 0; k< numHidden; k++)
	{
		change = hiddenDeltas[k] * trainInputs[patNum][j];
		changeHidden[j][k] = change;
		weightsIH[j][k] = weightsIH[j][k] + LR*change + M*changeHidden[j][k];
	}
}

int main()
{   
	srand((unsigned)time(0));   // Seed the random num generator
	initWeights();
	cout << "created weights" << endl;

	initData();
	cout << "created Training data and starting training" << endl;

	train();
	cout << "Finished training" << endl;

	//Training has finished.
	cout << "testing" << endl;
	test_1();
	
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	//----------------------------------------------------------------------
    return 0;
}

// Helper function for using CUDABackProp
//=============================NEEDS WORK==================================
cudaError_t CudaBackPropPrep(int numInputs,
							 double hiddenActivation[], int numHidden, 
							 double Hidden2Activation[], int numHidden2,
							 double outputActivation[], int numOutput,
							 double weightsIH[][], double weightsH1H2[][], double weightsHO[][], 
							 double outputDeltas[], double hiddenDeltas[], double hidden2Deltas[],
							 double changeHidden[][], double changeHidden2[][], double changeOutput[][]
							)
{
	cudaError_t cudaStatus;
	//---------------------device versions of vectors---------------------
    double *dev_HiddenActivation = 0;
    double *dev_Hidden2Activation = 0;
	double *dev_outputActivation = 0;

	double *dev_weightsIH = 0;
	double *dev_weightsH1H2 = 0;
	double *dev_weightsHO = 0;

	double *dev_outputDeltas = 0;
	double *dev_hiddenDeltas = 0;
	double *dev_hidden2Deltas = 0;

	double *dev_changeHidden = 0;
	double *dev_changeHidden2 = 0;
	double *dev_changeOutput = 0;
	//---------------------------------------------------------------------------

    // Choose which GPU to run on
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // -----------------------------------Allocate 1D GPU buffers for vectors-----------------------------------
    cudaStatus = cudaMalloc((void**)&dev_HiddenActivation, numHidden * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_Hidden2Activation, numHidden2 * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_OutputActivation, numOutput * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_outputDeltas, numOutput * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_HiddenDeltas, numHidden * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudaStatus = cudaMalloc((void**)&dev_Hidden2Deltas, numHidden2 * sizeof(double));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	//------------------------------------Allocate 2D GPU buffers for 2D vectors--------------------------------------
	//T* pElement = (T*)((char*)BaseAddress + Row * pitch) + Column; <---- addressing a "ptich"
	size_t pitch_WeightsIH;
	cudaStatus = cudaMallocPitch((void **)&dev_weightsIH,
		&pitch_WightsIH,
		numInputs * sizeOf(double),
		numHidden * sizeOf(double)
	)
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

	size_t pitch_WeightsH1H2;
	cudaStatus = cudaMallocPitch((void **)&dev_weightsH1H2,
		&pitch_WeightsH1H2,
		numHidden * sizeOf(double),
		numHidden2 * sizeOf(double)
		)
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

	size_t pitch_WeightsHO;
	cudaStatus = cudaMallocPitch((void **)&dev_weightsHO,
		pitch_WeightsHO,
		numHidden2 * sizeOf(double),
		numOutput * sizeOf(double)
		)
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

	cudaStatus = cudaMallocPitch((void **)&dev_weightsIH,
			size_t * pitch,
			size_t 	width,
			size_t 	height
		)
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

	size_t pitch_changeHidden;
	cudaStatus = cudaMallocPitch((void **)&dev_weightsIH,
			size_t * pitch,
			size_t 	width,
			size_t 	height
		)
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

	size_t pitch_changeHidden2;
	cudaStatus = cudaMallocPitch((void **)&dev_weightsIH,
			size_t * pitch,
			size_t 	width,
			size_t 	height
		)
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

	size_t pitch_changeOutput;
	cudaStatus = cudaMallocPitch((void **)&dev_weightsIH,
		size_t * pitch,
		size_t 	width,
		size_t 	height
	)
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

	//====================================================================================================================================

    //------------------------------------------Copy input vectors from host memory to GPU buffers---------------------------------------
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	//====================================================================================================================================
	return cudaStatus;
    
}
//--------------------------------------------------------
void train()
{
	cudaError_t cudaStatus = cudaBackPropprep();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cuda failed!");
		return 1;
	}
	for (int j = 0; j <= numEpochs; j++)
	{
		errorTotal = 0;
		for (int i = 0; i < numPatterns; i++)
		{
			patNum = i;
			//Calculate the output and error for this pattern.
			feedforward();
			errorTotal = errorTotal + RMSerror;
			
			//-----------------------Launch kernels on the GPU with one thread for each element in outside array----------------
			//replace size with length of outer array in nested loop.
			CudaBackProp_part1 <<<1, numOutput >>>();
			__syncthreads();
			CudaBackProp_part2 << <1, numHidden2 >>>();
			__syncthreads();
			CudaBackProp_part3 <<<1, numHidden >>>();
			__syncthreads();
			CudaBackProp_part4 <<<1, numHidden2 >>>();
			__syncthreads();
			CudaBackProp_part5 <<<1, numHidden >>>();
			__syncthreads();
			CudaBackProp_part6 << <1, numInputs >> >();
			__syncthreads();

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
				goto Error;
			}


		}
		errorTotal = sqrt(errorTotal / numPatterns);
		if (j % 100 == 0) { cout << "epoch = " << j << " Error = " << errorTotal << endl; }
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	//-------------------------Need to do this for all weights vectors-------------------------
	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//-----------------------------------------------------------------------------------------
	Error:
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);

	return cudaStatus;
	
}

void initWeights()
{
	// Initialize weights to random values.
	for (int j = 0; j < numHidden; j++)
	{
		for (int i = 0; i < numInputs - 1; i++)
		{
			weightsIH[i][j] = getRand(-.2, .2);

		}
		weightsIH[j][numInputs - 1] = 1;//Bias
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

void initData()
{
	//Training set of points inside/outside a circle
	for (int j = 0; j < numPatterns; j++)
	{
		for (int i = 0; i < numInputs; i++)
		{
			trainInputs[i][j] = getRand(minRange, maxRange);
		}

		if (((trainInputs[j][0] * trainInputs[j][0]) + (trainInputs[j][1] * trainInputs[j][1])) < (radius*radius))
		{//if in circle answer is 1
			trainOutput[j] = 1;
		}//otherwise answer is 0
		else { trainOutput[j] = 0; }
	}
}

void feedforward()
{
	//input nodes dont have activations, except an assignment from raw to an array

	//hidden nodes activations
	for (int i = 0; i < numHidden; i++)
	{
		SumOf = 0.0;
		for (int j = 0; j < numInputs; j++)
		{
			SumOf = SumOf + (trainInputs[patNum][j] * weightsIH[j][i]);
		}
		hiddenActivation[i] = sigmoid(SumOf);
	}

	for (int i = 0; i < numHidden2; i++)
	{
		SumOf = 0.0;
		for (int j = 0; j < numHidden; j++)
		{
			SumOf = SumOf + (hiddenActivation[j] * weightsH1H2[j][i]);
		}
		hidden2Activation[i] = sigmoid(SumOf);
	}

	//output nodes activations
	for (int i = 0; i < numOutput; i++)
	{
		SumOf = 0.0;
		for (int j = 0; j < numHidden2; j++)
		{
			SumOf = SumOf + (hidden2Activation[j] * weightsHO[j][i]);
		}
		outputActivation[i] = sigmoid(SumOf);
	}

	//Author Note:Needs general case for more than 1 output
	Guess = outputActivation[0];
	errThisPat = trainOutput[patNum] - Guess;
	RMSerror = errThisPat * errThisPat;

}

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
}

double getRand(double num1, double num2)
{
	return double((num1 - num2)*(rand() / double(RAND_MAX)) + num1); //(b-a)*random.random() + a
}

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

__global__ double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

__device__ double dsigmoid(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

void test_2()
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
}
