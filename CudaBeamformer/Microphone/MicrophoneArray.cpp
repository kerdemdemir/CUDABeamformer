#include "MicrophoneArray.h"


MicrophoneArray::MicrophoneArray()
{
	int arraySize = Config::getInstance().arraySize;
	float elemDistFromMid = arraySize/2;
	for (int i = 0; i < arraySize; i++)
	{
		if (arraySize % 2 == 0)
		{
			micropshoneList.emplace_back((float(i - elemDistFromMid) + 0.5) * Config::getInstance().distBetweenMics);
		}
		else
		{
			micropshoneList.emplace_back((float(i - elemDistFromMid)) * Config::getInstance().distBetweenMics);
		}
	}
}

MicrophoneArray::~MicrophoneArray()
{
}

void MicrophoneArray::InsertSound( const std::vector<float>& rawData, float focusDist, float steerAngle)
{
	for (auto& microphone : micropshoneList)
	{
		microphone.feed(rawData, focusDist, steerAngle, 0);
	}
}

void MicrophoneArray::Beamform(std::vector<float>& outputData, float focusDist, float steerAngle)
{
	std::fill(outputData.begin(), outputData.end(), 0.0);
	for (size_t i = 0; i < Config::getInstance().packetSize; i++)
	{
		for (auto& microphone : micropshoneList)
		{
			int delay = microphone.getDelay(focusDist, steerAngle) + Config::getInstance().getMicMaxDelay();
			outputData[i] += microphone.getData(delay + i);
		}
		outputData[i] /= Config::getInstance().arraySize;
	}
}