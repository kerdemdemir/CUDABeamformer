#pragma once

#include "../Config/Config.h"

using leapMap = std::unordered_map< int, std::shared_ptr<SharpVector>  >;
using leapIter = leapMap::iterator;

class Microphone
{
public:
	Microphone( float distCenter ) ;
	~Microphone();

	float getSteeringDelay(float steeringAngle) const
	{
		float returnVal = -m_distCenter * sin(steeringAngle * GLOBAL_PI / 180.0) / GLOBAL_SOUND_SPEED * (float)Config::getInstance().samplePerSecond;
		return returnVal;
	}

	float getFocusDelay(float focusDist) const
	{
		if (focusDist < 0.1)
			return 0;
		float returnVal = (pow(m_distCenter, 2) / (2.0 * focusDist)) / GLOBAL_SOUND_SPEED * (float)Config::getInstance().samplePerSecond;
		return returnVal;
	}

	float getDelay(float focusDist, float steeringAngle) const
	{
		return getFocusDelay(focusDist) + getSteeringDelay(steeringAngle);
	}


	void feed(const SharpVector& input, float focusDist, float steeringAngle, int speakerID)
	{
		size_t delay = getDelay(focusDist, steeringAngle) + Config::getInstance().getMicMaxDelay();
		auto beginIter = input.data();
		leapIter leapIte = getLeapIter(speakerID, delay);
		auto tempLeap = *leapIte->second;
		for (size_t k = 0; k < m_data.size() + delay; k++)
		{
			if (k < delay)
			{
				m_data[k] += tempLeap[k];
			}
			else if (k < m_data.size())
			{
				float soundData = *beginIter++;
				m_data[k] += soundData;
			}
			else
			{
				float soundData = *beginIter++;
				leapIte->second->at(k - m_data.size()) = soundData;
				m_leapTotalData[k - m_data.size()] += soundData;
			}
		}
	}

	leapIter getLeapIter(int speakerID, float delay)
	{
		leapIter leapIte = m_leapData.find(speakerID);
		if (leapIte == m_leapData.end())
		{
			leapIte = m_leapData.emplace(speakerID, std::make_shared<SharpVector>(delay)).first;
		}

		return leapIte;
	}


	void clearData()
	{
		std::fill(m_data.begin(), m_data.end(), 0);
		std::fill(m_leapTotalData.begin(), m_leapTotalData.end(), 0);
	}

	void clearLeapData()
	{
		m_leapData.clear();
	}

	const SharpVector& getData() const
	{
		return m_data;
	}

	const SharpVector& getLeapData() const
	{
		return m_leapTotalData;
	}

	float getData(size_t index) const
	{
		if (m_data.size() > index)
		{
			return m_data[index];
		}
		else
		{
			auto leapIndex = index - m_data.size();
			return m_leapTotalData[leapIndex];
		}
	}

private:

	float m_distCenter;
	std::unordered_map<int, std::shared_ptr<SharpVector>>  m_leapData; // Leap Data for each source
	SharpVector m_leapTotalData; // Leap Data for each source
	SharpVector m_data;
};
