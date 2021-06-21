// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"
#include "helper_image.h"
#include "helper_string.h"
#include "helper_timer.h"
#include "exception.h"

int BLOCKSIZE = 16;	        // number of threads in each block

int N = 2;

int NUMTRIALS = N * 1024;   // size of the array

// ranges for the random numbers:
const float GMIN = 20.0;    // ground distance in meters
const float GMAX = 30.0;    // ground distance in meters
const float HMIN = 10.0;    // cliff height in meters
const float HMAX = 40.0;    // cliff height in meters
const float DMIN = 10.0;    // distance to castle in meters
const float DMAX = 20.0;    // distance to castle in meters
const float VMIN = 30.0;    // initial cannonball velocity in meters / sec
const float VMAX = 50.0;    // initial cannonball velocity in meters / sec
const float THMIN = 70.0;   // cannonball launch angle in degrees
const float THMAX = 80.0;   // cannonball launch angle in degrees

// degrees-to-radians -- callable from the device:
__device__	float Radians(float d)
{
	return (M_PI / 180.f) * d;
}

// function prototypes:
void CudaCheckError();
void TimeOfDaySeed();
float Ranf(float, float);
int Ranf(int, int);

// the kernel:
__global__	void MonteCarlo(float* dvs, float* dths, float* dgs, float* dhs, float* dds, int* dhits)
{
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    const float GRAVITY = -9.8; // acceleration due to gravity in meters / sec^2
    const float TOL = 5.0;      // tolerance in cannonball hitting the castle in meters

	// randomize everything:
	float v		= dvs[gid];
	float thr	= Radians(dths[gid]);
	float vx	= v * cos(thr);
	float vy	= v * sin(thr);
	float g		= dgs[gid];
	float h		= dhs[gid];
	float d		= dds[gid];

	int numHits = 0;

	// see if the ball doesn't even reach the cliff:
	float t = -vy / (0.5 * GRAVITY);
	float x = vx * t;
	if (x > g)
	{
        // see if the ball hits the vertical cliff face:
        t = g / vx;
        float y = vy * t + 0.5 * GRAVITY * (t * t);
        if (y > h)
        {
            // the ball hits the upper deck:
            // the time solution for this is a quadratic equation of the form:
            // at^2 + bt + c = 0.
            // where 'a' multiplies time^2
            //       'b' multiplies time
            //       'c' is a constant
            float a = 0.5 * GRAVITY;
            float b = vy;
            float c = -h;
            float disc = b * b - 4.f * a * c;   // quadratic formula discriminant

            // successfully hits the ground above the cliff:
            // get the intersection:
            disc = sqrtf(disc);
            float t1 = (-b + disc) / (2.f * a);	// time to intersect high ground
            float t2 = (-b - disc) / (2.f * a);	// time to intersect high ground

            // only care about the second intersection
            float tmax = t1;
            if (t2 > t1)
                tmax = t2;

            // how far does the ball land horizontlly from the edge of the cliff?
            float upperDist = vx * tmax - g;

            // see if the ball hits the castle:
            if (fabs(upperDist - d) <= TOL)
            {
                numHits = 1;
            }
        }   // if ball clears the cliff face
	}   // if ball gets as far as the cliff face

	dhits[gid] = numHits;
}

// these two #defines are just to label things
// other than that, they do nothing:
#define IN
#define OUT


// main program:

int
main(int argc, char* argv[])
{
    if (argc >= 2)
        BLOCKSIZE = atoi(argv[1]);
    if (argc >= 3)
        NUMTRIALS = atoi(argv[2]) * 1024;

    int NUMBLOCKS = NUMTRIALS / BLOCKSIZE;

    TimeOfDaySeed();

    int dev = findCudaDevice(argc, (const char**)argv);

    // better to define these here so that the rand() calls don't get into the thread timing:
    float* hvs = new float[NUMTRIALS];
    float* hths = new float[NUMTRIALS];
    float* hgs = new float[NUMTRIALS];
    float* hhs = new float[NUMTRIALS];
    float* hds = new float[NUMTRIALS];
    int* hhits = new int[NUMTRIALS];

    // fill the random-value arrays:
    for (int i = 0; i < NUMTRIALS; i++)
    {
        hvs[i] = Ranf(VMIN, VMAX);
        hths[i] = Ranf(THMIN, THMAX);
        hgs[i] = Ranf(GMIN, GMAX);
        hhs[i] = Ranf(HMIN, HMAX);
        hds[i] = Ranf(DMIN, DMAX);
    }

    // allocate device memory:
    float   *dvs, *dths, *dgs, *dhs, *dds;
    int     *dhits;

    cudaMalloc(&dvs, NUMTRIALS * sizeof(float));
    cudaMalloc(&dths, NUMTRIALS * sizeof(float));
    cudaMalloc(&dgs, NUMTRIALS * sizeof(float));
    cudaMalloc(&dhs, NUMTRIALS * sizeof(float));
    cudaMalloc(&dds, NUMTRIALS * sizeof(float));
    cudaMalloc(&dhits, NUMTRIALS * sizeof(int));
    CudaCheckError();

    // copy host memory to the device:
    cudaMemcpy(dvs, hvs, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dths, hths, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dgs, hgs, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dhs, hhs, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dds, hds, NUMTRIALS * sizeof(float), cudaMemcpyHostToDevice);
    CudaCheckError();

    // setup the execution parameters:
    dim3 grid(NUMBLOCKS, 1, 1);
    dim3 threads(BLOCKSIZE, 1, 1);

    // allocate cuda events that we'll use for timing:
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    CudaCheckError();

    // let the gpu go quiet:
    cudaDeviceSynchronize();

    // record the start event:
    cudaEventRecord(start, NULL);
    CudaCheckError();

    // execute the kernel:
    MonteCarlo <<< grid, threads >>> (IN dvs, IN dths, IN dgs, IN dhs, IN dds, OUT dhits);

    // record the stop event:
    cudaEventRecord(stop, NULL);
    CudaCheckError();

    // wait for the stop event to complete:
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    CudaCheckError();

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    CudaCheckError();

    // compute the performance
    double secTotal = msecTotal / 1000;
    double megaTrialsPerSecond = (double)NUMTRIALS / secTotal / 1000000.;

    // copy result from the device to the host:
    cudaMemcpy(hhits, dhits, NUMTRIALS * sizeof(int), cudaMemcpyDeviceToHost);
    CudaCheckError();

    // add up the hhits[ ] array: :
    int numHits = 0;
    for (int i = 0; i < NUMTRIALS; i++)
    {
        numHits += hhits[i];
    }

    // compute the probability:
    float probability = (float)numHits / (float)(NUMTRIALS);

    // print the result:
    fprintf(stderr, "%2d threads/block : %8d trials ; probability = %6.2f%% ; megatrials/sec = %6.2lf\n", 
            BLOCKSIZE, NUMTRIALS, 100. * probability, megaTrialsPerSecond);

    // clean up host memory:
    delete [] hvs;
    delete [] hths;
    delete [] hgs;
    delete [] hhs;
    delete [] hds;
    delete [] hhits;

    // clean up device memory:
    cudaFree(dvs);
    cudaFree(dths);
    cudaFree(dgs);
    cudaFree(dhs);
    cudaFree(dds);
    cudaFree(dhits);
    CudaCheckError();

    return 0;
}

void
CudaCheckError()
{
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
    {
        fprintf(stderr, "CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
    }
}

// call this if you want to force your program to use
// a different random number sequence every time you run it:
void
TimeOfDaySeed()
{
    struct tm y2k = { 0 };
    y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
    y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

    time_t  timer;
    time(&timer);
    double seconds = difftime(timer, mktime(&y2k));
    unsigned int seed = (unsigned int)(1000. * seconds);    // milliseconds
    srand(seed);
}

float
Ranf(float low, float high)
{
    float r = (float)rand();                // 0 - RAND_MAX
    float t = r / (float)RAND_MAX;          // 0. - 1.

    return   low + t * (high - low);
}

int
Ranf(int ilow, int ihigh)
{
    float low = (float)ilow;
    float high = ceil((float)ihigh);

    return (int)Ranf(low, high);
}