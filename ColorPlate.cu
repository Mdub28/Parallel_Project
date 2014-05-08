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
#include <thrust/device_vector.h>
#include "CycleTimer.h"
using namespace cv; 
using namespace std;

__global__ void 
gaussian_blur(
	unsigned char *  original,
	unsigned char *  result,
	int width,
	int height,
	float *  kernel) 
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if( x%2 == 0 || y %2==0 ||y <= 0 || y >= height -1 || x<=0 || x>=width-1) {
		return;
	} else {
		float total= 0.f;
		total += 1.f / 16 *static_cast<float>(original[x + -1 + width * (y + -1)]);
		total += 2.f / 16 *static_cast<float>(original[x + -1 + width * (y + 0)]);
		total += 1.f / 16 *static_cast<float>(original[x + -1 + width * (y + 1)]);
		total += 2.f / 16 *static_cast<float>(original[x + 0 + width * (y + -1)]);
		total += 4.f/16 *static_cast<float>(original[x + 0 + width * (y + 0)]);
		total += 2.f / 16 *static_cast<float>(original[x + 0 + width * (y + 1)]);
		total += 1.f / 16 *static_cast<float>(original[x + 1 + width * (y + -1)]);
		total += 2.f / 16 *static_cast<float>(original[x + 1 + width * (y + 0)]);
		total += 1.f / 16 *static_cast<float>(original[x + 1 + width * (y + 1)]);
		result[(x/2) + (y/2)*(width/2)] = static_cast<unsigned char>(total); 
  }
}


__global__ void ssd (
	unsigned char * temp,
	unsigned char * image,
	float * result,
	int leftbor,
	int rightbor,
	int topbor,
	int botbor,
	int width,
	int xoff,
	int yoff)  
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x < leftbor || x >= rightbor || y < topbor || y >=botbor) {
		return; 
	} else {
		result[x-leftbor+(y-topbor)*(width/2)] = (fabsf((float)(temp[x+y*width]-image[x+xoff+(y+yoff)*width])));
	}
}

	

__global__ void shift_image(
int x, 
int y, 
unsigned char* original,
unsigned char* result, 
int width, 
int height) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int index = ix + iy * (width);
	if (ix >= width || iy >= height) {
		return;
	}
	
	//check x bound has not shifted to another row
	if (ix + x < 0 || ix + x >= width) {
		return;
	}
	//check y bound has not gone out of bounds
	if (iy + y < 0 || iy + y >= height) {
		return;
	}
	
	result[ix + x + (iy+y)*width] = original[index];
}

			
struct square {
	__host__ __device__ float operator() (const float& a) const {
		return pow(a,2);
	}
};

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
		cudaMemcpy(blue.data, dblue, height*blue.step, cudaMemcpyDeviceToHost);
		cudaMemcpy(green.data, dgreen, height*green.step, cudaMemcpyDeviceToHost);
		cudaMemcpy(red.data, dred, height*red.step, cudaMemcpyDeviceToHost);
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
		printf("offsets are %d %d %d %d \n", offsets[0], offsets[1], offsets[2], offsets[3]);
		return;
	} else {
		unsigned char* bresult;
		unsigned char* gresult;
		unsigned char* rresult;
		float * gdif;
		float * rdif;
		size_t resultSize = (height/2) * (width/2) * sizeof(unsigned char);
		size_t ssdSize = (height/2) * (width/2) * sizeof(float);
		cudaMalloc(&bresult, resultSize);
		cudaMalloc(&gresult, resultSize);
		cudaMalloc(&rresult, resultSize);
		cudaMalloc(&gdif, ssdSize );
		cudaMalloc(&rdif, ssdSize);
		
		dim3 pixBlockDim(32, 16);
		dim3 pixGridDim((width + pixBlockDim.x -1) / pixBlockDim.x,
			(height + pixBlockDim.y -1) / pixBlockDim.y);
		gaussian_blur<<<pixGridDim, pixBlockDim>>>(dblue, bresult, width, height, dkernel );
		gaussian_blur<<<pixGridDim, pixBlockDim>>>(dgreen, gresult, width, height, dkernel );
		gaussian_blur<<<pixGridDim, pixBlockDim>>>(dred, rresult, width, height, dkernel);
		int * roffsets = new int[4];
		findOffsetCuda(bresult, gresult, rresult, dkernel, height/2, width/2, roffsets);
		
		float rlowest = -1;
		float glowest = -1;	
		int leftbor = width / 4;
		int rightbor = width - leftbor;
		int topbor = height/4;
		int botbor = height - topbor;

		float * bglist = new float[((height/2) *(width/2))];	
		float * brlist = new float[((height/2) *(width/2))];	
		for (int i=-2; i <=2; i++) {
			for (int j=-2; j <=2; j++) {
				ssd<<<pixGridDim, pixBlockDim>>>(dblue, dgreen, gdif, 
					leftbor, rightbor, topbor, botbor, width, i+2*roffsets[0], j+2*roffsets[1]);
				ssd<<<pixGridDim, pixBlockDim>>>(dblue, dred, rdif, 
					leftbor, rightbor, topbor, botbor, width, i+2*roffsets[2], j+2*roffsets[3]);	
				cudaMemcpy(&bglist[0], gdif, ssdSize, cudaMemcpyDeviceToHost);
				cudaMemcpy(&brlist[0], rdif, ssdSize, cudaMemcpyDeviceToHost);
				thrust::device_vector<float> bgdif(bglist, &bglist[(height/2) * (width/2) ]);
				thrust::device_vector<float> brdif(brlist, &brlist[(height/2) * (width/2) ]);
				thrust::transform(bgdif.begin(), bgdif.end(), bgdif.begin(), square());
				thrust::transform(brdif.begin(), brdif.end(), brdif.begin(), square());
				float gssd = thrust::reduce(bgdif.begin(), bgdif.end(), (float) 0, thrust::plus<float>());
				float rssd = thrust::reduce(brdif.begin(), brdif.end(), (float) 0, thrust::plus<float>());	
				
				if (rssd < rlowest || rlowest == -1) {
					offsets[2] = i+2*roffsets[2];
					offsets[3] = j+2*roffsets[3];
					rlowest = rssd;
				}
				if (gssd <glowest || glowest == -1) { 
					offsets[0] = i+2*roffsets[0];
					offsets[1] = j+2*roffsets[1];
					glowest = gssd;
				}
			}
		}
		printf("offsets are %d %d %d %d \n", offsets[0], offsets[1], offsets[2], offsets[3]);
		cudaFree(bresult);
		cudaFree(gresult);
		cudaFree(rresult);
		cudaFree(gdif);
		cudaFree(rdif); 
		delete[] roffsets;
		delete[] bglist;
		delete[] brlist;
	}
}
	
Mat runCUDA(Mat image) {
	int wbor = image.cols / 20;
	int hbor = image.rows / 20;
	Mat blue = image(Rect(wbor, hbor, image.cols - 2*wbor, image.rows/3 - 2*hbor)).clone();
	Mat green = image(Rect(wbor, image.rows/3 + hbor, image.cols - 2*wbor, image.rows/3 - 2*hbor)).clone();
	Mat red = image(Rect(wbor, 2 * image.rows/3 + hbor, image.cols - 2*wbor, image.rows/3 - 2*hbor)).clone();
	
	int height = blue.rows;
	int width = blue.cols;
	size_t channelSize = height*blue.step; 
	unsigned char* dblue;
	unsigned char* dred;
  unsigned char* dgreen;
	float * dkernel;
	float kernel[9] = { 1.f / 16 ,2.f / 16,1.f/16 ,2.f/16,4.f/16,2.f/16,1.f/16,2.f/16,1.f/16};
	int * offsets = new int[4];
	cudaMalloc( &dblue, channelSize);
	cudaMalloc( &dgreen, channelSize);
	cudaMalloc( &dred, channelSize);
	cudaMalloc(&dkernel, 9*sizeof(float));
	cudaMemcpy(dblue, blue.data, channelSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dgreen, green.data, channelSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dred, red.data, channelSize, cudaMemcpyHostToDevice);		
	cudaMemcpy(dkernel, &kernel[0], 9*sizeof(float), cudaMemcpyHostToDevice);
	findOffsetCuda(dblue, dgreen, dred, dkernel, height, width, offsets);
	

	//cudaMemcpy(fred.data, rresult, resultSize, cudaMemcpyDeviceToHost);
/*
	Mat origmerge;  
	std::vector<Mat> omergevec;
	omergevec.push_back(fblue);
	omergevec.push_back(fgreen);
	omergevec.push_back(fred);
	merge(omergevec, origmerge);
*/
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
	cudaFree(dblue);
	cudaFree(dgreen);
	cudaFree(dred);
	cudaFree(dkernel);
	delete [] offsets;
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
		return 0;
	}
	
	Mat final;
	double startTime = CycleTimer::currentSeconds();
	if (type=="s") {
		final = runCUDA(image);
		//take_input(image);
	} else if (type =="p") {
		return 0;
	} else {
		return 0;
	}
	double endTime = CycleTimer::currentSeconds();
	printf("total time: %.3f ms \n", 1000.f * (endTime-startTime));
	
	//save and display image
	//namedWindow("Merged image", WINDOW_AUTOSIZE);
	//imshow("Merged image", final);
	//imwrite("result.jpg", final);
	return 0;
}