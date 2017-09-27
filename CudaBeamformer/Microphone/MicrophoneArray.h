#pragma once

#include "Microphone.h"

class MicrophoneArray
{
public:
	MicrophoneArray();
	~MicrophoneArray();

	void InsertSound( const std::vector<double>& rawData, double focusDist, double steerAngle ); 
	void Beamform(std::vector<double>& outputData, double focusDist, double steerAngle);

	std::vector<Microphone> micropshoneList;
};

