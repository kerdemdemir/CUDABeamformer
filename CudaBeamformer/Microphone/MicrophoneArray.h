#pragma once

#include "Microphone.h"

class MicrophoneArray
{
public:
	MicrophoneArray();
	~MicrophoneArray();

	void InsertSound( const std::vector<float>& rawData, float focusDist, float steerAngle ); 
	void Beamform(std::vector<float>& outputData, float focusDist, float steerAngle);

	std::vector<Microphone> micropshoneList;
};

