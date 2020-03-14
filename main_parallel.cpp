/*
	main.cpp
	Converts a RGB image to greyscale using GPU and parallel programming
*/

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#define block_width 32
#define cuda_check_errors(val) check( (val), #val, __FILE__, __LINE__)

using namespace cv;
using namespace std;

// Placehold for images when opened.
Mat image_rgba;
Mat image_grey;

/*
	Imports from the kernel.cu
*/
template<typename T>
void check(T err, const char* const func, const char* const file, 
			const int line);
void rgba_to_grey_launcher(uchar4 *const d_rgba, unsigned char *const d_grey,
							size_t rows, size_t cols);

size_t num_rows(Mat image) { return image.rows; }
size_t num_cols(Mat image) { return image.cols; }
size_t num_pixels(Mat image) { return image.cols * image.rows; }

/*
	Opens the image from the given directory and if it was ok, 
	returns it in opencv format (Mat).
*/
void pre_process_RGB(const string &in_file) {
	Mat image;
	image = imread(in_file.c_str(), CV_LOAD_IMAGE_COLOR);
	if (image.empty() || !image.data) {
		cerr << "Couldn't open file: " << in_file << endl;
		exit(1);
	}

	cvtColor(image, image_rgba, CV_BGR2RGBA);

	if (!image_rgba.isContinuous()) {
		cerr << "RGBA Image isn't continuous!! Exiting." << endl;
		exit(1);
	}
}

/*
	Returns the greyscale image placehold from the given RGBA image.
*/
void pre_process_grey() {
	image_grey.create(num_rows(image_rgba), num_cols(image_rgba), CV_8UC1);
	if (!image_grey.isContinuous()) {
		cerr << "Grey image isn't continuous!! Exiting." << endl;
		exit(1);
	}
}

/*
	Does the image creation.
	Allocates memory for the images in GPU.
	Copies the RGBA image from CPU to GPU (host to device).
*/
void pre_process(uchar4 **h_rgba, unsigned char **h_grey,
                 uchar4 **d_rgba, unsigned char **d_grey,
                 const string &in_file) {
	cuda_check_errors(cudaFree(0));

	pre_process_RGB(in_file);

	pre_process_grey();

	*h_rgba = (uchar4 *) image_rgba.ptr<unsigned char>(0);
	*h_grey = image_grey.ptr<unsigned char>(0);

	const size_t np = num_pixels(image_rgba);
	cuda_check_errors(cudaMalloc(d_rgba, sizeof(uchar4) * np));
	cuda_check_errors(cudaMalloc(d_grey, sizeof(unsigned char) * np));
	cuda_check_errors(cudaMemset(*d_grey, 0, sizeof(unsigned char) * np));
	cuda_check_errors(cudaMemcpy(*d_rgba, *h_rgba, sizeof(uchar4) * np, 
									cudaMemcpyHostToDevice));
}

/*
	Outputs the result into the output file.
*/
void post_process(const string &out_file, unsigned char *h_grey,
					size_t rows, size_t cols) {
	Mat output(rows, cols, CV_8UC1, (void *) h_grey);
	imwrite(out_file.c_str(), output);
}


int main(int argc, char **argv) {
    uchar4 *h_rgba, *d_rgba;
    unsigned char *h_grey, *d_grey;

    string in_file;
    string out_file;

    in_file = string(argv[1]);
    if (argc == 2) 
        out_file = "output.png";
    else if (argc == 3)
        out_file = argv[2];
    else {
        cerr << "Usage: ./RGB_grey input_file [output_filename]" << endl;
        exit(1);
    }

    pre_process(&h_rgba, &h_grey, &d_rgba, &d_grey, in_file);
    
    clock_t start = clock();
    rgba_to_grey_launcher(d_rgba, d_grey,
    						num_rows(image_rgba), num_cols(image_rgba));
    cudaDeviceSynchronize();
    cuda_check_errors(cudaGetLastError());

    cuda_check_errors(cudaMemcpy(h_grey, d_grey, 
    					sizeof(unsigned char) * num_pixels(image_rgba), 
    					cudaMemcpyDeviceToHost));

    float elapsed = 1000 * double(clock() - start) / CLOCKS_PER_SEC;
    printf("Runtime: %.3f miliseconds. \n", elapsed);
    
    post_process(out_file, h_grey, num_rows(image_rgba), num_cols(image_rgba));
    
    return 0;
}