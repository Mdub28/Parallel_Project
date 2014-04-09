#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


int main(int argc, char** argv) {
	string input; 
	string output;
	
	if (argc > 2) {
		input = string(argv[1]);
		output = string(argv[2]);
	}
	
  cv::Mat image = imread(input, CV_LOAD_IMAGE_GREYSCALE); 
	cv::Mat blue = image(Rect(0, 0, image.cols, image.rows/3));
	cv::Mat green = image(Rect(0, image.rows/3, image.cols, image.rows/3));
	cv::Mat red = image(Rect(0, 2 * image.rows/3, image.cols, image.rows/3));

}