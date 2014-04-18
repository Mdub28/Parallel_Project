#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

int main(int argc, char** argv) {
	string input; 
	string output;
	
	if (argc > 2) {
		input = string(argv[1]);
		output = string(argv[2]);
	}
	
  Mat image = imread(input, CV_LOAD_IMAGE_GREYSCALE);

  #trim and get each negative
	int wbor = image.cols / 20;
	int hbor = image.rows / 20;
	Mat blue = image(Rect(wbor, hbor, image.cols - 2*wbor, image.rows/3 - 2*hbor));
	Mat green = image(Rect(wbor, image.rows/3 + hbor, image.cols - 2*wbor, image.rows/3 - 2*hbor));
	Mat red = image(Rect(wbor, 2 * image.rows/3 + hbor, image.cols - 2*wbor, image.rows/3 - 2*hbor));

	int * offsets = new int[4]();
	findOffset(blue, green, red, offsets);
	
	#shift images by offset
	shiftImage(-offsets[0], offsets[1], green);
	shiftImage(-offsets[2], offsets[3], red);
	
	#merge 
	Mat final;
	std::vector<Mat> mergevec;
	mergevec.push_back(blue);
	mergevec.push_back(green);
	mergevec.push_back(red);
	merge(mergevec, final);
	
	#final trim
	wbor = final.cols /20;
	hbor = final.rows / 20;
	Mat blue = final(Rect(wbor, hbor, final.cols - 2*wbor, final.rows - 2*hbor));
	
	#save and display image
	namedWindow("Merged image", WINDOW_AUTOSIZE);
	imshow("Merged image", final);
	#imwrite(output, color);
}

void findOffset(Mat blue, Mat green, Mat red, int * offsets) {
	if (blue.rows < 500 && column < 500) {
		int wbor = blue.cols / 6;
		int hbot = blue.rows / 6;
		Mat filter = blue(Rect(wbor, hbor, final.cols - 2*wbor, final.rows - 2*hbor));
		Mat filteredG;
		Mat filteredR;
		double gminval, gmaxval, rminval, rmaxval;
		Point gmin, gmax, rmin, rmax;
		
		matchTemplate(green, filter, filteredG, CV_TM_CCORR_NORMED);
		matchTemplate(red, filter, filteredR, CV_TM_CCORR_NORMED);
		minMaxLoc(filteredG, &gminval, &gmaxval, &gmin, &gmax);
		minMaxLoc(filteredR, &rminval, &rmaxval, &rmin, &rmax);
		
		offsets[0] = gmax.x - filteredG.cols/2;
		offsets[1] = gmax.y - filteredG.rows/2;
		offsets[2] = rmax.x - filteredR.cols/2;
		offsets[3] = rmax.y - filteredR.rows/2; 
	} else {
		Mat smallB, smallG, smallR, redSub, greenSub;
		pyrDown(blue, smallB);
		pyrDown(green, smallG);
		pyrDown(red, smallR);
		int * roffsets = new int[4];
		findOffset(smallB, smallG, smallR, roffsets);
		int rlowest = -1;
		int glowest = -1;
		
		int wbor = blue.cols / 6;
		int hbot = blue.rows / 6;
		Mat bluemid = blue(Rect(wbor, hbor, blue.cols - 2*wbor, blue.rows - 2*hbor));
		for (int i=-1; i <=1; i++) {
			for (int j=-1; j<=1; j++) {
				Mat greenmid = green(Rect(wbor+i+2*roffsets[0], hbor+j+2*roffsets[1], blue.cols - 2*wbor, blue.rows - 2*hbor));
				Mat redmid = red(Rect(wbor+i+2*roffsets[2], hbor+j+2*roffsets[3], blue.cols - 2*wbor, blue.rows - 2*hbor));
				
				subtract(bluemid, redmid, redSub);
				subtract(bluemid, greenmid, greenSub);
				pow(redSub, 2, smallR);
				pow(greenSyb, 2, smallG);
				
				int redSSD = sum(smallR).val[0]; 
				int greenSSD = sum(smallG).val[0];
				
				if (redSSD < rlowest || rlowest == -1) {
					offsets[2] = i+2*roffsets[2];
					offsets[3] = j+2*roffsets[3]
					rlowest = redSSD;
				}
				if (greenSSD <glowest || glowest == -1) }
					offsets[0] = i+2*roffsets[0];
					offsets[1] = j+2*roffsets[1]
					glowest = greenSSD;
				}
			}
		}
	}
}

void applyTransformation(Mat kernel, Mat image, int repeats) {
	while (repeats > 0) {
		filter2D(image, image, -1, kernel, Point(-1, -1), 0, BORDER_CONSTANT);
		repeats--;
	}
	
	return;
}

void shiftImage(int lr, int ud, Mat image) {
	left_kernel = Mat::zeros(3, 3, CV_32F);
	up_kernel = Mat::zeros(3, 3, CV_32F);
	left_kernel.at<float>(0, 1) = 1.0f;
	up_kernel.at<float>(1, 0) = 1.0f;
	
	if (lr < 0) {
		applyTransformation(left_kernel, image, -lr);
	} else {
		applyTransformation(cv::flip(left_kernel, left_kernel, 1), image, -lr);
	}
	
	if (ud > 0) {
		applyTransformation(up_kernel, image, ud);
	} else {
		applyTransformation(cv::flip(up_kernel, up_kernel, 0), image, -ud);
	}
}