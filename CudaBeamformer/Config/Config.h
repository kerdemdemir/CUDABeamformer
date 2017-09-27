#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

using SharpVector = std::vector<double>;

#define GLOBAL_PI 3.14159265
#define GLOBAL_SOUND_SPEED 34300

class Config
{
public:
	static Config& getInstance()
	{
		static Config    instance;
		return instance;
	}

	int getMicMaxDelay() const
	{
		double totalMicLen = (Config::getInstance().arraySize * Config::getInstance().distBetweenMics - 1) / 2.0;
		return totalMicLen / GLOBAL_SOUND_SPEED * (double)Config::getInstance().samplePerSecond;
	}

	size_t samplePerSecond;
	size_t arraySize;
	size_t distBetweenMics;
	size_t packetSize;
private:
	Config() 
	{

	}
	Config(Config const&);         // Don't Implement.
	void operator=(Config const&); // Don't implement

};

