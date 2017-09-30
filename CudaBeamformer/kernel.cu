
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
float get_time()
{
	LARGE_INTEGER t, f;
	QueryPerformanceCounter(&t);
	QueryPerformanceFrequency(&f);
	return (float)t.QuadPart / (float)f.QuadPart;
}

#else

#include <sys/time.h>
#include <sys/resource.h>

float get_time()
{
	struct timeval t;
	struct timezone tzp;
	gettimeofday(&t, &tzp);
	return t.tv_sec + t.tv_usec*1e-6;
}

#endif

struct MicParamsGpu
{
	float *outputData;
	float *rawData;
	int    packetSize;
	float *leapData;
	int    leapStride;
	int    *delays;
	int    arraySize;
	int    stride;
};

cudaError_t beamformWithCudaHelper(MicrophoneArray& array, SharpVector& outputData);

__device__ float GetElement(MicParamsGpu params, int curMic, int index ) {	
	if (params.packetSize > index)
	{
		return params.rawData[curMic*params.packetSize + index];
	}
	else
	{
		int leapIndex = (index - params.packetSize) + curMic*params.leapStride;
		return params.leapData[leapIndex];
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
			curVal = GetElement(params, i, currentStartIndex + k + params.delays[i]);
		}
		params.outputData[currentStartIndex + k] = curVal;
	}
}

int createPulse(SharpVector& data, size_t readSize, float sampleRate)
{
	float f0 = 1500;
	float ts = 1.0 / sampleRate;
	//float vz = 3000;
	for (size_t i = 0; i < readSize; i++)
	{
		float realTime = i  * ts;
		float realPart = cos(2.0*GLOBAL_PI*realTime*f0);
		data.push_back(realPart);

	}
	return data.size();
}

int main()
{
	Config& ins = Config::getInstance();

	ins.samplePerSecond = 44000;
	ins.arraySize = 32;
	ins.distBetweenMics = 10;
	ins.packetSize = 44000;

	MicrophoneArray array;
	SharpVector rawData;
	createPulse(rawData, 44000, 44000);
	array.InsertSound(rawData, 1000, 45);

	float startTime = get_time();

	SharpVector outputData(Config::getInstance().packetSize);
	array.Beamform(outputData, 1000, 45); // CPU toplama 

	
	
	//beamformWithCudaHelper(array, outputData); // GPU toplama 
	float endTime = get_time();
	std::cout << "CPU Time spent: " << endTime - startTime;
	
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t beamformWithCudaHelper(MicrophoneArray& array, SharpVector& outputData)
{	
	cudaError_t cudaStatus;
	MicParamsGpu params;
	params.arraySize = array.micropshoneList.size();
	params.packetSize = Config::getInstance().packetSize;
	params.leapStride = Config::getInstance().getMicMaxDelay()*2;
	cudaMalloc(&params.rawData, array.micropshoneList.size() * sizeof(float) * params.packetSize);
	cudaMalloc(&params.leapData, array.micropshoneList.size() * sizeof(float) * params.leapStride);
	cudaMalloc(&params.delays, array.micropshoneList.size() * sizeof(int));

	
	cudaMalloc(&params.outputData, Config::getInstance().packetSize * sizeof(float) );
	std::vector<int> delayVec;
	params.stride = params.packetSize / 1000;
	cudaStatus = cudaGetLastError();
	for (int i = 0; i < params.arraySize; i++)
	{
		cudaMemcpy( params.rawData + i * params.packetSize, array.micropshoneList[i].getData().data(),
						params.packetSize* sizeof(float), cudaMemcpyHostToDevice);
		cudaStatus = cudaGetLastError();
		cudaMemcpy(params.leapData + i * params.leapStride, array.micropshoneList[i].getLeapData().data(),
						params.leapStride* sizeof(float), cudaMemcpyHostToDevice);
		cudaStatus = cudaGetLastError();
		delayVec.push_back(array.micropshoneList[i].getDelay(1000, 45) + Config::getInstance().getMicMaxDelay());

	}
	cudaStatus = cudaGetLastError();
	cudaMemcpy(params.delays, delayVec.data(), delayVec.size() * sizeof(int), cudaMemcpyHostToDevice);

	float startTime = get_time();
	// Launch a kernel on the GPU with one thread for each element.
	beamformKernel2 << <1, 1000 >> >(params);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.


	cudaMemcpy(outputData.data(), params.outputData, Config::getInstance().packetSize * sizeof(float), cudaMemcpyDeviceToHost);
	float endTime = get_time();
	std::cout << "CPU Time spent: " << endTime - startTime;
 	return cudaStatus;
}