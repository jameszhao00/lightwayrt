#include "Stats.h"
#include "windows.h"

Stats Stats::two_way( int samples_per_iteration, int num_bouces, int width, int height )
{
	Stats stats;
	stats.num_combinations = 
		width * height *
		samples_per_iteration * (1 /* eye path */ + num_bouces + 1 /* light path */);
	return stats;
}

Stats Stats::pt( int num_cycles, int width, int height )
{
	Stats stats;
	stats.num_combinations = 
		width * height *
		num_cycles;
	return stats;
}
void Stats::start()
{
	QueryPerformanceCounter((LARGE_INTEGER*)&start_counter);
}

void Stats::stop()
{
	LARGE_INTEGER end_counter;
	QueryPerformanceCounter(&end_counter);
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	duration_in_seconds = (double)(end_counter.QuadPart - start_counter) / freq.QuadPart;
}

double Stats::combinations_per_second() const
{
	return num_combinations / duration_in_seconds;
}
