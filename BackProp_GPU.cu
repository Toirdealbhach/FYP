#pragma comment(lib,"cublas.lib") 
#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h> 
#include <cuda_runtime.h>
#include <time.h>
#include "cublas_v2.h"
#include <curand.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"




using namespace std;

//==============================Function Prototypes================================
double getRand();
__global__ void initWeights(float *dst, unsigned int seed);
__global__ void sigmoid(float * dst);
__global__ void deltaCalcOutput(float *OutActivation, float *Outputdelta, float *targets);
__global__ void deltaCalcHidden(float *Activation, float *delta);
__global__ void weightUpdate(float *d_W, float *d_D, float *d_N);
__global__ void printGuess(float *output);
__global__ void transferInput(float *src,float **dst);


class NeuralNet {
	int numInputs;
	int num_layers;
	
	int numOutput;
	int patNum = 0;

	const float LR_IH = 0.1f;      
	const float LR_HO = 0.1f;
	
	float errThisPat = 0.0f;
	float outPred = 0.0f;                   // "Expected" output values.
	float RMSerror = 0.0f;// Root Mean Squared error.
	
	//==================Matrices Vital for Backprop================
	int *layer_sizes;

	float **d_W; //weights
	float **d_N; //nodes
	float **d_D; //deltas
	float **temp_error;
	float *dev_trainInputs;

	float **trainInputs;
	float **trainOutput; 

	public:
	NeuralNet(int num_layers);
	void feedforward();
	void displayResults( int numPatterns);
	void train(int numEpochs, int numPatterns);
	void backPropagate(); //Needs to be written
	void initData(int numPatterns, int numInputs, int numOutput);
};
	
NeuralNet::NeuralNet(int num_layer){

	cudaError_t cudaStat;
	num_layers = num_layer;
	layer_sizes = new int[num_layers];
	for (int i = 0; i<num_layers; i++) {
		cout << "Enter number of nodes in layer" << i + 1 << endl;
		cin >> layer_sizes[i];
	}
	numInputs = layer_sizes[0];
	numOutput = layer_sizes[num_layers];
	d_W = new float *[num_layers-1];
	d_N = new float *[num_layers];
	d_D = new float *[num_layers];

	
	for (int count = 0; count < num_layers - 1; count++)
	{
		cudaStat = cudaMalloc((void **)(d_W + count), layer_sizes[count] * layer_sizes[count + 1] * sizeof(float)); // device
	}
	for (int count = 0; count < num_layers; count++)
	{
		cudaStat = cudaMalloc((void **)(d_N + count), layer_sizes[count] * 1 * sizeof(float)); // device
		cudaStat = cudaMalloc((void **)(d_D + count), layer_sizes[count] * 1 * sizeof(float)); // device
	}
	
	
	int data_length = 1024;
	int block = 256, grid = data_length / block;  //define block and grid
	for (int count = 0; count < num_layers- 1; count++) {
		initWeights<<<grid, block>>>(d_W[count], (unsigned)time(NULL));
	}
	cudaMalloc((void**)&dev_trainInputs, layer_sizes[0] * sizeof(float));
	
}

int IDX2C(int i, int j, int ld) {
	return (((j)*(ld)) + (i));
}
int main() {
	srand((unsigned)time(NULL)); // Seed the generator with system time.
	const int numPatterns = 4;      // Input patterns for XOR experiment.
	const int numEpochs = 1;
	const int numInputs = 3; 
	const int numOutput = 1;
	clock_t t1, t2;
	
	
	cout << "Enter number of layers" << endl;
	int num_layers;
	cin >> num_layers;
	NeuralNet NN(num_layers);

	NN.initData(numPatterns, numInputs, numOutput);
	

	t1 = clock();
	NN.train(numEpochs,numPatterns);
	//NN.displayResults(numPatterns);
	t2 = clock();
	float diff((float)t2 - (float)t1);
	cout << endl;
	cout << diff << endl;

	return EXIT_SUCCESS;
}

void NeuralNet::feedforward() {

	cudaError_t cudaStatus; // cudaMalloc status
	cublasStatus_t stat; // CUBLAS functions status
	cublasHandle_t handle;
//	cudaMalloc((void**)&handle, sizeof(cublasHandle_t));
	  // CUBLAS context

	stat = cublasCreate(&handle); // initialize CUBLAS context
	//stat = cublasSetVector(layer_sizes[0], sizeof(float), trainInputs[patNum], 1, d_N[0], layer_sizes[0]);
	//cout << *trainInputs[patNum] << endl;
	//cout << *trainInputs[patNum] + 1 << endl;
	cudaStatus = cudaMemcpy(d_N[0], trainInputs[patNum], (layer_sizes[0] -1) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Get last error returned error code %d after copying input vector in feedforward function!\n", stat);
	}
	int data_length;
	int block;
	int grid;
	
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Get last error returned error code %d after copying input vector in feedforward function!\n", stat);
	}
	block = 256;  //define block and grid
	float al = 1.0f; // al =1
	float bet = 0.0f; // beta is 0. reference the documnentation===================================================
	for (int count = 0; count < num_layers - 1; count++) {
		cublasSgemv(handle, CUBLAS_OP_N, layer_sizes[count], layer_sizes[count + 1], &al, d_W[count], layer_sizes[count], d_N[count], 1, &bet, d_N[count + 1], 1);
		if (stat != cudaSuccess) {
			fprintf(stderr, "Get last error returned error code %d after launching matrix vector mul in feedforward function!\n", stat);
		}
		data_length = layer_sizes[count + 1]; grid = 1; block = data_length;
		if (data_length > 256) {
			block = 256;
			grid = (data_length / block);
		}//BASED ON ASSUMPTION THAT DATA LENGTH = n*256;
		sigmoid<<<grid, block>>>(d_N[count + 1]);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Get last error returned error code %d after launching feedforward sigmoid kernel!\n", cudaStatus);
		}
	}
	//cublasDestroy(handle);
}

void NeuralNet::backPropagate(){
	cudaError_t cudaStatus; // cudaMalloc status
	cublasStatus_t stat; // CUBLAS functions status
	cublasHandle_t handle; // CUBLAS context
	stat = cublasCreate(&handle); // initialize CUBLAS context
	int data_length;
	int block = 32;
	int grid;  //define block and grid
	float al = 1.0f; // al =1
	float bet = 0.0f; // beta is 0. reference the documnentation===================================================
	for (int count = num_layers-1; count > 0; count--) {
		data_length = layer_sizes[count];
		grid = 1; block = data_length;
		if (data_length > 256) {
			block = 256;
			grid = (data_length / block);
		}
		if (count == num_layers-1) {
			//BASED ON ASSUMPTION THAT DATA LENGTH = n*256;

			deltaCalcOutput<<<grid, block>>>(d_N[count], d_D[count], trainOutput[patNum]);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Get last error returned error code %d after launching backprop deltaCalcOutput!\n", cudaStatus);
			}
		}
		else{
			stat = cublasSgemv(handle, CUBLAS_OP_N, layer_sizes[count], layer_sizes[count - 1], &al, d_W[count], layer_sizes[count], d_D[count], 1, &bet, d_D[count - 1], 1);
						
			if (stat != cudaSuccess) {
				fprintf(stderr, "Get last error returned error code %d after launching matrix vector mul in backprop function!\n", cudaStatus);
			}
			deltaCalcHidden <<<grid, block>>> (d_N[count], d_D[count]);
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "Get last error returned error code %d after launching backprop deltaCalcHidden!\n", cudaStatus);
			}
		}
	}
	dim3 GRID, BLOCK;
	for (int count = num_layers - 1; count > 0; count--){
		data_length = layer_sizes[count] ;
		GRID.x = 1; BLOCK.x = data_length;
		if (data_length > 256) {
			BLOCK.x = 256;
			GRID.x = (data_length / BLOCK.x);
		}
		data_length = layer_sizes[count+1];
		GRID.y = 1; BLOCK.y = data_length;
		if (data_length > 256) {
			BLOCK.y = 256;
			GRID.y = (data_length / BLOCK.y);
		}
		weightUpdate <<<GRID, BLOCK>>> ( d_W[count],  d_D[count+1],  d_N[count-1]);
	}

	//cublasDestroy(handle);
}

void NeuralNet::train(int numEpochs, int numPatterns) {
	// Train the network.
	for (int j = 0; j < 100; j++) {
		RMSerror = 0;
		for (int i = 0; i < 4; i++) {

			//Select a pattern at random.
			patNum = rand() % numPatterns;

			feedforward();
			//cudaDeviceSynchronize();
			//backPropagate();
			//cudaDeviceSynchronize();
			//RMSerror = RMSerror + abs(backPropagate(patNum, trainOutput[patNum]));
		}
		//calcOverallError(trainInputs, numPatterns, trainOutput);

		//Display the overall network error after each epoch
		cout << "epoch = " << j << " RMS Error = " << RMSerror << endl;
	}
}

void NeuralNet::initData(int numPatterns, int numInputs, int numOutput) {
	
	if (layer_sizes[num_layers-1] != numOutput){
		cout << "Dataset and network architecture are not compatible. Try adjusting output layer sizes or training output data" << endl;
		goto Error;
	}
	else if (layer_sizes[0] != numInputs) {
		cout << "Dataset and network architecture are not compatible. Try djusting input layer sizes or training input data" << endl;
		goto Error;
	}
	trainInputs = new float*[numPatterns];
	trainOutput = new float*[numPatterns];
	for (int i = 0; i < numPatterns; i++){
		trainInputs[i] = new float[numInputs];
		trainOutput[i] = new float[numOutput];
	}

	
	trainInputs[0][0] =1.0f;
	trainInputs[0][1] = 0.0f;
	trainInputs[0][2] = 1.0f; // Bias
	trainOutput[0][0] = 0.0f;

	trainInputs[1][0] = 0.0f;
	trainInputs[1][1] = 1.0f;
	trainInputs[1][2] = 1.0f; // Bias
	trainOutput[1][0] = 1.0f;

	trainInputs[2][0] = 1.0f;
	trainInputs[2][1] = 1.0f;
	trainInputs[2][2] = 1.0f; // Bias
	trainOutput[2][0] = 0.0f;

	trainInputs[3][0] = 0.0f;
	trainInputs[3][1] = 0.0f;
	trainInputs[3][2] = 1.0f; // Bias
	trainOutput[3][0] = 1.0f;

Error:
	//throw;
}

void NeuralNet::displayResults(int numPatterns) {
	for (int i = 0; i <= numPatterns; i++) {
		patNum = i;
		feedforward();
		cout << "pat = " << patNum + 1 <<
			" actual = " << trainOutput[patNum][0] <<
			" neural model = " << endl;
		printGuess <<<1,1>>>(d_N[num_layers - 1]);
	}
}

double getRand() {
	return double(rand() / double(RAND_MAX));
}
//THE ELEMWISE SIGMOID FUNCTION
__global__ void sigmoid(float * dst){
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	dst[n] = 1/(1+exp(-dst[n]));
}

__global__ void deltaCalcOutput(float *OutActivation, float *Outputdelta, float *targets){	
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	float error = targets[n] - OutActivation[n];
	Outputdelta[n] = error * (1 / (1 + exp(-OutActivation[n]))*(1 - 1 / (1 + exp(-OutActivation[n]))));
}

__global__ void deltaCalcHidden(float *Activation,float *delta){
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	delta[n] = delta[n] * (1 / (1 + exp(-Activation[n]))*(1 - 1 / (1 + exp(-Activation[n]))));
}

__global__ void weightUpdate(float *d_W,float *d_D,float *d_N){
	int2 pos;
	pos.x = blockIdx.x*blockDim.x + threadIdx.x;//row j
	pos.y = blockIdx.y*blockDim.y + threadIdx.y;//column k
	int n = pos.y*blockDim.y*gridDim.y + pos.x;
	float N = 0.1;
	d_W[n] = d_W[n] + N*d_D[pos.x] * d_N[pos.y];
}


__global__ void initWeights(float *dst, unsigned int seed){
	//params are: seed,sequence num,offset,handle
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	dst[n] = dst[n]/(float)(seed);
	while(dst[n] > 5) { 
		dst[n]=dst[n]/2; 
	}
	if (n%(seed % 3) == 0) {
		dst[n] = dst[n] * -1;
	}
}

__global__ void transferInput(float *src,float **dst) {
	int2 pos;
	pos.x = blockIdx.x*blockDim.x + threadIdx.x;//row j
	pos.y = blockIdx.y*blockDim.y + threadIdx.y;//column k
	int n = pos.y*blockDim.x*gridDim.x + pos.x;
	dst[pos.y][pos.x] = src[n];
}

__global__ void printGuess(float *output){
	printf("%f \n",output[0]);
}

__global__ void printError(float *output,float *target) {
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	float error = target[n] - output[n];
	printf("%f \n", error );
}
