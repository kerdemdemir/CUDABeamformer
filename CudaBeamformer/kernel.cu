
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
//#include <sndfile.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "Config\Config.h"
#include "Microphone\MicrophoneArray.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <algorithm>
#include <cstdlib>

#ifdef WIN32

#include <windows.h>
double get_time()
{
	LARGE_INTEGER t, f;
	QueryPerformanceCounter(&t);
	QueryPerformanceFrequency(&f);
	return (double)t.QuadPart / (double)f.QuadPart;
}

#else

#include <sys/time.h>
#include <sys/resource.h>

double get_time()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec*1e-6;
}

#endif

struct MicParamsGpu
{
	double *outputData;
	double **rawData;
	int    packetSize;
	double **leapData;
	int    *delays;
	int    arraySize;
	int    stride;
};

cudaError_t beamformWithCudaHelper(MicrophoneArray& array, SharpVector& outputData);

__device__ double GetElement(MicParamsGpu params, int curMic, int index ) {	
	if (params.packetSize > index)
	{
		return params.rawData[curMic][index];
	}
	else
	{
		int leapIndex = index - params.packetSize;
		return params.leapData[curMic][leapIndex];
	}
}

__global__ void beamformKernel2(MicParamsGpu params)
{
	int xIndex = threadIdx.x;
	int currentStartIndex = xIndex * params.stride;

	for (int k = 0; k < params.stride; k++)
	{
		float curVal = 0;
		for (int i = 0; i < params.arraySize; i++)
		{
			curVal += params.delays[i];// GetElement(params, i, currentStartIndex + params.delays[i]);
		}
		params.outputData[currentStartIndex + k] = curVal;
	}
}

int createPulse(SharpVector& data, size_t readSize, double sampleRate)
{
	double f0 = 1500;
	double ts = 1.0 / sampleRate;
	double vz = 3000;
	for (size_t i = 0; i < readSize; i++)
	{
		double realTime = i  * ts;
		double realPart = cos(2.0*GLOBAL_PI*realTime*f0) *
			exp(-1.0 * ((i - readSize / 2) * (i - readSize / 2)) / (vz * vz));


		data.push_back(realPart);

	}
	return data.size();
}

int main()
{
	Config& ins = Config::getInstance();

	ins.samplePerSecond = 44100;
	ins.arraySize = 32;
	ins.distBetweenMics = 10;
	ins.packetSize = 44100;

	MicrophoneArray array;
	SharpVector rawData;
	createPulse(rawData, 88200, 44100);
	array.InsertSound(rawData, 1000, 45);

	double startTime = get_time();

	SharpVector outputData(Config::getInstance().packetSize);
	//array.Beamform(rawData, 1000, 45);

	
	
	beamformWithCudaHelper(array, outputData);
	double endTime = get_time();
	std::cout << "CPU Time spent: " << endTime - startTime;
	
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t beamformWithCudaHelper(MicrophoneArray& array, SharpVector& outputData)
{	
	MicParamsGpu params;

	cudaMalloc(&params.rawData, array.micropshoneList.size());
	cudaMalloc(&params.leapData, array.micropshoneList.size());
	cudaMalloc(&params.delays, array.micropshoneList.size());

	params.packetSize = Config::getInstance().packetSize;
	cudaMalloc(&params.outputData, Config::getInstance().packetSize);
	params.arraySize = array.micropshoneList.size();
	params.stride = params.packetSize / 1000;
	for (int i = 0; i < array.micropshoneList.size(); i++ )
	{
		thrust::device_vector<double>* inputData = new  thrust::device_vector<double>(array.micropshoneList[i].getData());
		thrust::device_vector<double>* leapData = new  thrust::device_vector<double>(array.micropshoneList[i].getLeapData());
		double *inputRaw = thrust::raw_pointer_cast(inputData->data());
		double *leapRaw = thrust::raw_pointer_cast(leapData->data());
		params.rawData[i]  = inputRaw;
		params.leapData[i] = leapRaw;
		params.delays[i] = array.micropshoneList[i].getDelay(1000, 45) + Config::getInstance().getMicMaxDelay();

	}

	cudaError_t cudaStatus;

	// Launch a kernel on the GPU with one thread for each element.
	beamformKernel2 << <1, 1000 >> >(params);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.


	cudaMemcpy(outputData.data(), params.outputData, outputData.size(), cudaMemcpyDeviceToHost);

	return cudaStatus;
}