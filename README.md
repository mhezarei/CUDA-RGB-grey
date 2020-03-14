# RGB-grey

This project is intented to convert a RGB image to a greyscale one using parallel programming which results into ~100-200x (exactly 203x speed for the sample.jpg included here) speedup comparing to a serial program doing the same conversion.

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
$ nvcc kernel.cu main.cpp -o RGB_grey_parallel `pkg-config opencv --cflags --libs`
```
And run it using this (and to see the execution time):
```
$ ./RGB_grey_parallel sample.jpg [Output file]
```
If the output file is not specified, then the result is saved into "output.png".

You could use this to run the serial program which does the same thing:
```
$ g++ kernel.cu main.cpp -o RGB_grey_serial `pkg-config opencv --cflags --libs`; ./RGB_grey_serial sample.jpg [Output file]
```
Again, the result will be saved into "output.png" if not specified.

## Contributing
Anything helpful is much appreciated!
