# RGB-grey

This project is intented to convert a RGB image to a greyscale one using parallel programming which results into ~100-200x (0.5ms for parallel version and 107ms for serial version which results into ~203x speedup tested on the sample.jpg included in the repo) speedup comparing to a serial program doing the same conversion.

For a guide about parallel programming you could refer to [CUDA C++](http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

## Prerequisites

First you have to install CUDA for you Nvidia graphic card. You can install the suitable version from [CUDA Download](https://developer.nvidia.com/cuda-downloads).

You also need to install opencv for ubuntu using the following command:

```
$ sudo apt-get install libopencv-dev
```

## Usage

Compile the parallel program using the following command:
```
$ nvcc kernel.cu main_parallel.cpp -o RGB_grey_parallel `pkg-config opencv --cflags --libs`
```
And run it using this (and to see the execution time):
```
$ ./RGB_grey_parallel sample.jpg [Output file]
```
If the output file is not specified, then the result is saved into "output.png".

You could use this to run the serial program which does the same thing:
```
$ g++ main_serial.cpp -o RGB_grey_serial `pkg-config opencv --cflags --libs`; ./RGB_grey_serial sample.jpg [Output file]
```
Again, the result will be saved into "output.png" if not specified.

## Contributing
Anything helpful is much appreciated!
