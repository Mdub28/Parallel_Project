#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <algorithm>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
using namespace cv; 
using namespace std;

void findOffsetCuda(
	unsigned char* blue,
	unsigned char* green,
	unsigned char* red,
	int height,
	int width) {
	return;
	}
	
Mat runCUDA(Mat image) {
	int wbor = image.cols / 20;
	int hbor = image.rows / 20;
	Mat blue = image(Rect(wbor, hbor, image.cols - 2*wbor, image.rows/3 - 2*hbor));
	Mat green = image(Rect(wbor, image.rows/3 + hbor, image.cols - 2*wbor, image.rows/3 - 2*hbor));
	Mat red = image(Rect(wbor, 2 * image.rows/3 + hbor, image.cols - 2*wbor, image.rows/3 - 2*hbor));
	int channelSize = blue.cols*blue.rows*sizeof(unsigned char);
	unsigned char* dblue;
	unsigned char* dred;
  unsigned char* dgreen;
	cudaMalloc(&dblue, channelSize);
	cudaMalloc(&dgreen, channelSize);
	cudaMalloc(&dred, channelSize);
	cudaMemcpy(dblue, blue.ptr<unsigned char>(0), channelSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dgreen, green.ptr<unsigned char>(0), channelSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dred, red.ptr<unsigned char>(0), channelSize, cudaMemcpyHostToDevice);
	printf("doneee \n");		
			/*
	//Get separate channels
	double startTime = CycleTimer::currentSeconds();
	getChannels<<<pixGridDim, pixBlockDim>>>(original, blue, green, red, ch, cw, hbor, wbor);
	double endTime = CycleTimer::currentSeconds();
	printf("Sequential trim and separate: %.3f ms \n", 1000.f * (endTime-startTime));
	
	
	//Find offset
	findOffsetCuda(blue, green, red, ch, cw);
	return image;
	*/
	return image;
}

int main(int argc, char** argv) {
	string input; 
	string output;
	string type;

	if (argc > 3) {
	  type = string(argv[1]);
		input = string(argv[2]);
		output = string(argv[3]);
	}
	
	Mat image = imread(input, 0); 
	if (image.empty()) {
		return -1;
	}
	
	Mat final;
	double startTime = CycleTimer::currentSeconds();
	if (type=="s") {
		final = runCUDA(image);
	} else if (type =="p") {
		return -1;
	} else {
		return -1;
	}
	double endTime = CycleTimer::currentSeconds();
	printf("total time: %.3f ms \n", 1000.f * (endTime-startTime));
	
	//save and display image
	//namedWindow("Merged image", WINDOW_AUTOSIZE);
	//imshow("Merged image", final);
	imwrite("result.jpg", final);
}