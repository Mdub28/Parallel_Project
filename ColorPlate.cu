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
#include <thrust/host_vector.h>

#include "CycleTimer.h"
using namespace cv; 
using namespace std;

__global__ void gaussian_blur(
	unsigned char * original,
	unsigned char * result,
	int width,
	int height,
	float * kernel) 
{
	int y	= blockIdx.y * blockDim.y + threadIdx.y;		// current row
	int x	= blockIdx.x * blockDim.x + threadIdx.x;		// current column
	
	if(y%2 == 0 || x%2 == 0|| y == 0 || y >= height -1 || x==0 || x>=width-1) {
		return;
	} else {
		float total= 0;
		for(int i=-1; i <= 1; i++) {
			for(int j=-1; j<=1; j++) {
				int index = x + i + width * (y + j);
				float pixel = static_cast<float>(original[index]);
				total += pixel * kernel[i+1 + (j+1)*3];
			}
		}
		result[(x / 2) + (y/2)*(width/2)] = static_cast<unsigned char>(total);
	}
}

void findOffsetCuda(
	unsigned char * dblue,
	unsigned char * dgreen,
	unsigned char * dred,
	float * dkernel,
	int height,
	int width,
	int * offsets) 
{
	if (height < 200 && width < 200) {
		Mat blue = Mat::Mat(height, width, CV_8UC1);
		Mat green = Mat::Mat(height, width, CV_8UC1);
		Mat red = Mat::Mat(height, width, CV_8UC1);
		cudaMemcpy(blue.ptr<unsigned char>(0), dblue, height*width*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaMemcpy(green.ptr<unsigned char>(0), dgreen, height*width*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		cudaMemcpy(red.ptr<unsigned char>(0), dred, height*width*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		int wbor = width / 10;
		int hbor = height / 10;
		Mat filter = blue(Rect(wbor, hbor, blue.cols - 2*wbor, blue.rows - 2*hbor));
		Mat filteredG;
		Mat filteredR; 
		double gminval, gmaxval, rminval, rmaxval;
		Point gmin, gmax, rmin, rmax;
		
		matchTemplate(green, filter, filteredG, CV_TM_CCORR_NORMED);
		matchTemplate(red, filter, filteredR, CV_TM_CCORR_NORMED);
		minMaxLoc(filteredG, &gminval, &gmaxval, &gmin, &gmax);
		minMaxLoc(filteredR, &rminval, &rmaxval, &rmin, &rmax);
		offsets[0] = gmax.x - wbor;
		offsets[1] = gmax.y - hbor;
		offsets[2] = rmax.x - wbor;
		offsets[3] = rmax.y - hbor; 
		return;
	} else {
		unsigned char* bresult;
		unsigned char* gresult;
		unsigned char* rresult;
		int resultSize = height /2 * width /2 * sizeof(unsigned char);
		cudaMalloc(&bresult, resultSize);
		cudaMalloc(&gresult, resultSize);
		cudaMalloc(&rresult, resultSize);
		
		dim3 pixBlockDim(32, 32);
		dim3 pixGridDim((width + pixBlockDim.x -1) / pixBlockDim.x,
			(height + pixBlockDim.y -1) / pixBlockDim.y);
		gaussian_blur<<<pixGridDim, pixBlockDim>>>(dblue, bresult, width, height, dkernel );
		gaussian_blur<<<pixGridDim, pixBlockDim>>>(dgreen, gresult, width, height, dkernel );
		gaussian_blur<<<pixGridDim, pixBlockDim>>>(dred, rresult, width, height, dkernel);
		
		int * roffsets = new int[4];
		findOffsetCuda(bresult, gresult, rresult, dkernel, height/2, width/2, roffsets);
		
		/*
		int rlowest = -1;
		int glowest = -1;	
		int wbor = blue.cols / 6;
		int hbor = blue.rows / 6;
		Mat bluemid = blue(Rect(wbor, hbor, blue.cols - 2*wbor, blue.rows - 2*hbor));
		for (int i=-2; i <=2; i++) {
			for (int j=-2; j<=2; j++) {
				Mat greenmid = green(Rect(wbor+i+2*roffsets[0], hbor+j+2*roffsets[1], blue.cols - 2*wbor, blue.rows - 2*hbor));
				Mat redmid = red(Rect(wbor+i+2*roffsets[2], hbor+j+2*roffsets[3], blue.cols - 2*wbor, blue.rows - 2*hbor));
				 
				subtract(bluemid, redmid, redSub);
				subtract(bluemid, greenmid, greenSub);
				pow(redSub, 2, smallR);
				pow(greenSub, 2, smallG);
				
				int redSSD = sum(smallR).val[0]; 
				int greenSSD = sum(smallG).val[0];
				
				if (redSSD < rlowest || rlowest == -1) {
					offsets[2] = i+2*roffsets[2];
					offsets[3] = j+2*roffsets[3];
					rlowest = redSSD;
				}
				
				if (greenSSD <glowest || glowest == -1) {
					offsets[0] = i+2*roffsets[0];
					offsets[1] = j+2*roffsets[1];
					glowest = greenSSD;
				}
			}
		}
		*/
	}
}
	
Mat runCUDA(Mat image) {
	int wbor = image.cols / 20;
	int hbor = image.rows / 20;
	Mat blue = image(Rect(wbor, hbor, image.cols - 2*wbor, image.rows/3 - 2*hbor));
	Mat green = image(Rect(wbor, image.rows/3 + hbor, image.cols - 2*wbor, image.rows/3 - 2*hbor));
	Mat red = image(Rect(wbor, 2 * image.rows/3 + hbor, image.cols - 2*wbor, image.rows/3 - 2*hbor));
	float kernel[9] = { 1.f / 16 ,2.f / 16,1.f/16 ,2.f/16,4.f/16,2.f/16,1.f/16,2.f/16,1.f/16};
	for (int i=0; i < 9; i++) {
		printf("%f \n", kernel[i]);
	}
	int channelSize = blue.cols*blue.rows*sizeof(unsigned char);
	unsigned char* dblue;
	unsigned char* dred;
  unsigned char* dgreen;
	float * dkernel;
	int * offsets = new int[4];
	cudaMalloc(&dblue, channelSize);
	cudaMalloc(&dgreen, channelSize);
	cudaMalloc(&dred, channelSize);
	cudaMalloc(&dkernel, 9*sizeof(float));
	cudaMemcpy(dblue, blue.ptr<unsigned char>(0), channelSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dgreen, green.ptr<unsigned char>(0), channelSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dred, red.ptr<unsigned char>(0), channelSize, cudaMemcpyHostToDevice);		
	cudaMemcpy(dkernel, &kernel[0], 9*sizeof(float), cudaMemcpyHostToDevice);
	//findOffsetCuda(dblue, dgreen, dred, dkernel, blue.cols, blue.rows, offsets);
	
	unsigned char* bresult;
	unsigned char* gresult;
	unsigned char* rresult;
	int height = blue.rows;
	int width = blue.cols;
	int resultSize = height /2 * width /2 * sizeof(unsigned char);
	cudaMalloc(&bresult, resultSize);
	cudaMalloc(&gresult, resultSize);
	cudaMalloc(&rresult, resultSize);
	
	dim3 pixBlockDim(32, 32);
	dim3 pixGridDim((width + pixBlockDim.x -1) / pixBlockDim.x,
		(height + pixBlockDim.y -1) / pixBlockDim.y);
	gaussian_blur<<<pixGridDim, pixBlockDim>>>(dblue, bresult, width, height, dkernel );
	gaussian_blur<<<pixGridDim, pixBlockDim>>>(dgreen, gresult, width, height, dkernel );
	gaussian_blur<<<pixGridDim, pixBlockDim>>>(dred, rresult, width, height, dkernel);
	
	Mat fblue = Mat::Mat(height/2, width/2, CV_8UC1);
	Mat fgreen = Mat::Mat(height/2, width/2, CV_8UC1);
	Mat fred = Mat::Mat(height/2, width/2, CV_8UC1);
	cudaMemcpy(fblue.ptr<unsigned char>(0), bresult, (height/2)*(width/2)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(fgreen.ptr<unsigned char>(0), gresult, (height/2)*(width/2)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(fred.ptr<unsigned char>(0), rresult, (height/2)*(width/2)*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	/*
	Mat origmerge; 
	std::vector<Mat> omergevec;
	omergevec.push_back(fblue);
	omergevec.push_back(fgreen);
	omergevec.push_back(fred);
	merge(omergevec, origmerge);
	*/
	imwrite("gaussed1.jpg", fblue);
	
				/*
	thrust::device_vector<unsigned char> blueVec(blue.ptr<unsigned char>(0), (--blue.end<unsigned char>()).ptr);
	thrust::device_vector<unsigned char> greenVec(green.ptr<unsigned char>(0), (--green.end<unsigned char>()).ptr);
	thrust::device_vector<unsigned char> redVec(red.ptr<unsigned char>(0), (--red.end<unsigned char>()).ptr);
	
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
	//imwrite("result.jpg", final);
}