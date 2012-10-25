#pragma once

struct Stats
{
	int num_combinations;
	static Stats two_way(int samples_per_iteration, int num_bouces,
		int width, int height);
	__int64 start_counter;
	void start();
	void stop();
	double duration_in_seconds;
	double combinations_per_second() const;
};