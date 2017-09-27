#include "MicrophoneArray.h"


MicrophoneArray::MicrophoneArray()
{
	int arraySize = Config::getInstance().arraySize;
	double elemDistFromMid = arraySize/2;
	for (int i = 0; i < arraySize; i++)
	{
		if (arraySize % 2 == 0)
		{
			micropshoneList.emplace_back((double(i - elemDistFromMid) + 0.5) * Config::getInstance().distBetweenMics);
		}
		else
		{
			micropshoneList.emplace_back((double(i - elemDistFromMid)) * Config::getInstance().distBetweenMics);
		}
	}
}

MicrophoneArray::~MicrophoneArray()
{
}

void MicrophoneArray::InsertSound( const std::vector<double>& rawData, double focusDist, double steerAngle)
{
	for (auto& microphone : micropshoneList)
	{
		microphone.feed(rawData, focusDist, steerAngle, 0);
	}
}

void MicrophoneArray::Beamform(std::vector<double>& outputData, double focusDist, double steerAngle)
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