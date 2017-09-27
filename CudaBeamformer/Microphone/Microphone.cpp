#include "Microphone.h"


Microphone::Microphone( double distCenter )
{
	m_distCenter = distCenter;
	m_data.resize(Config::getInstance().packetSize, 0);
	m_leapTotalData.resize(Config::getInstance().getMicMaxDelay() * 2, 0);
}


Microphone::~Microphone()
{
}
