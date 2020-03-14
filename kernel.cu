/*
	kernel.cu
	Holds the kernel for the main program
*/

#include <iostream>

#define BLOCK_WIDTH 32
#define cuda_check_errors(val) check( (val), #val, __FILE__, __LINE__)

using namespace std;

/*
	Reports the location of the occured error and exits the program
*/
template<typename T>
void check(T err, const char* const func, const char* const file, 
			const int line) {
	if (err != cudaSuccess) {
		cerr << "CUDA error at: " << file << ":" << line << endl;
		cerr << cudaGetErrorString(err) << " " << func << endl;
		exit(1);
	}
}

/*
	The primary kernel (heart of the program!)
	Each pixel p in the RGBA image is a struct of four unsigned chars:
		- p.x Which is the red channel number.
		- p.y The green channel.
		- p.z The blue channel.
		- p.w The alpha channel (which we ignore).
	For each greyscale pixel to be created we calculate this formula:
		p = .299*p.x + .587*p.y + .114*p.z
	The output is a single char because we only have one channel.
	
	In the kernel each thread is responsible for calculating the mentioned
	formula for each pixel and the put the result into the greyscale
	image placehold.
	First we find out where the thread (pixel) is and then we do the 
	above.
*/
__global__
void rgba_to_grey(uchar4 *const d_rgba, unsigned char *const d_grey, 
					size_t rows, size_t cols) {
	size_t j = blockIdx.y * blockDim.y + threadIdx.y;
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows || j >= cols) 
		return;
	
	uchar4 p = d_rgba[i * cols + j];
	d_grey[i * cols + j] = (unsigned char) (0.299f * p.x + 0.587f * p.y + 0.114f * p.z);
}

/*
	The image is divided into number of blocks.
	Each block holds BLOCK_WIDTH*BLOCK_WIDTH threads and in total we have
	(rows/BLOCK_WIDTH)*(cols/BLOCK_WIDTH) blocks.
*/
void rgba_to_grey_launcher(uchar4 *const d_rgba, unsigned char *const d_grey,
							size_t rows, size_t cols) {
    const dim3 block_size (BLOCK_WIDTH, BLOCK_WIDTH, 1);
    unsigned int grid_x = (unsigned int) (rows / BLOCK_WIDTH + 1);
    unsigned int grid_y = (unsigned int) (cols / BLOCK_WIDTH + 1);
    const dim3 grid_size (grid_x, grid_y, 1);
    rgba_to_grey<<<grid_size, block_size>>>(d_rgba, d_grey, rows, cols);
    cudaDeviceSynchronize();
    cuda_check_errors(cudaGetLastError());
}