#include <stdio.h>
#include <algorithm>
#include <vector>

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "CycleTimer.h"
using namespace cv; 

void setup(Mat image) {

}
__global__ void getChannels(
  unsigned char* original,
	unsigned char* blue,
	unsigned char* green,
	unsigned char* red,
	int height,
	int width,
	int hbor,
	int wbor) {
	int y	= blockIdx.y * blockDim.y + threadIdx.y;	
	int x	= blockIdx.x * blockDim.x + threadIdx.x;
	
	if (y >= height || x < wbor || x >= width - wbor) {
		return;
	}
	
	int channel = y / (height / 3);
	int newy = y - channel* (y/3);
	if (newy < hbor || newy >= height / 3 - hbor) {
		return;
	}	else if (channel == 0) {
		blue[(x - wbor) + newy * (width - 2*wbor)] = original[x + y * width];
	} else if (channel == 1) {
		green[(x - wbor) + newy * (width - 2*wbor)] = original[x + y * width];
	} else if (channel == 2) {
		red[(x - wbor) + newy * (width - 2*wbor)] = original[x + y * width];
	}	
}

void findOffsetCuda(
	unsigned char* blue,
	unsigned char* green,
	unsigned char* red,
	int height,
	int width) {
	
	}
	
Mat runCUDA(Mat image) {
  //Setup
	int sqrtBlockSize = 32;
	dim3 pixBlockDim(sqrtBlockSize, sqrtBlockSize);
	dim3 pixGridDim((image->width + pixBlockDim.x -1) / pixBlockDim.x,
			(image->height + pixBlockDim.y -1) / pixBlockDim.y);

	int h = image.rows;
	int w = image.cols;
	int ch =((image.rows / 3) * 9 / 10);
	int cw = (image.cols * 9 / 10);
	int hbor = (image.row / 3) / 20;
	int wbor = image.cols / 20;
	int channelSize = ch * cw * sizeof(unsigned char);
	int originalSize = image.rows * image.cols * sizeof(unsigned char);
	int finalSize = (ch * 9 / 10) * (cw * 9 / 10) * sizeof(unsigned char);
	
	unsigned char* blue, green, red, original, final;
	cudaMalloc(&blue, channelSize);
	cudaMalloc(&green, channelSize);
	cudaMalloc(&red, channelSize);
	cudaMalloc(&original, originalSize);
	cudaMalloc(&final, finalSize);	
	cudaMemcpy(original, image.ptr<unsigned char>(0), originalSize, cudaMemcpyHostToDevice);
			
	//Get separate channels
	double startTime = CycleTimer::currentSeconds();
	getChannels<<<pixGridDim, pixBlockDim>>>(original, blue, green, red, ch, cw, hbor, wbor);
	double endTime = CycleTimer::currentSeconds();
	printf("Sequential trim and separate: %.3f ms \n", 1000.f * (endTime-startTime));
	
	
	//Find offset
	findOffsetCuda(blue, green, red, ch, cw);
	//
	/*
	int * offsets = new int[4]();
	startTime = CycleTimer::currentSeconds();
	findOffset(blue, green, red, offsets);
	endTime = CycleTimer::currentSeconds();
	printf("Sequential findOffset: %.3f ms \n", 1000.f * (endTime-startTime));
	
	//shift images by offset
	startTime = CycleTimer::currentSeconds();
	shiftImage(-offsets[0], offsets[1], green);
	shiftImage(-offsets[2], offsets[3], red);
	endTime = CycleTimer::currentSeconds();
	printf("Sequential shift: %.3f ms \n", 1000.f * (endTime-startTime));
	
	//merge 
	Mat final;
	std::vector<Mat> mergevec;
	mergevec.push_back(blue);
	mergevec.push_back(green);
	mergevec.push_back(red);
	merge(mergevec, final);
	
	//final trim
	wbor = final.cols /20;
	hbor = final.rows / 20;
	return final(Rect(wbor, hbor, final.cols - 2*wbor, final.rows - 2*hbor));
	*/
	return image;
}