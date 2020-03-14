/*
	main_serial.cpp
*/
#include <iostream>
#include <string>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	Mat RGBImage;

	string in_file;
    string out_file;

    in_file = string(argv[1]);
    if (argc == 2) {
        out_file = "Output.png";
    } 
    else if (argc == 3) {
        out_file = argv[2];
    }
    else {
        cerr << "Usage: ./RGB_grey_serial input_file [output_filename]" << endl;
        exit(1);
    }

	RGBImage = imread(in_file);
	Mat grayScaleImage(RGBImage.size(), CV_8UC1);

	clock_t start = clock();
	cvtColor(RGBImage,grayScaleImage,CV_RGB2GRAY);
	
	printf("Runtime: %.3f miliseconds. \n", 1000 * double(clock() - start) / CLOCKS_PER_SEC);

	imwrite(out_file.c_str(), grayScaleImage);

	return 0;
}