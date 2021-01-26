#ifndef PIXELWORLD_H_
#define PIXELWORLD_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define W 1024
#define H 1024

#define N_BLOCKS 1024
#define N_THREADS 1024

#define BYTES_PER_COLOR 3

#define UNDEFINED 0
#define AIR 1
#define SAND 2
#define WATER 3
#define METAL 4

#define SETPIXEL(p, x, y, t) { \
		p[((y)*W + (x)) * 3] = (t == AIR) ? 0x00 : (t == SAND) ? 0xFF : (t == WATER) ? 0x00 : (t == METAL) ? 0x77 : 0x00;\
		p[((y)*W + (x)) * 3 + 1] = (t == AIR) ? 0x00 : (t == SAND) ? 0xFF : (t == WATER) ? 0x00 : (t == METAL) ? 0x77 : 0x00;\
		p[((y)*W + (x)) * 3 + 2] = (t == AIR) ? 0x00 : (t == SAND) ? 0x00 : (t == WATER) ? 0xFF : (t == METAL) ? 0x77 : 0x00; }

#define GETPIXEL(p, r, g, b, x, y) {\
		r = p[((y)*W + (x)) * 3];\
		g = p[((y)*W + (x)) * 3 + 1];\
		b = p[((y)*W + (x)) * 3 + 2]; }

#define UPDATEPIXEL(dst, src, x, y) {\
		dst[((y)*W + (x)) * 3] = src[((y)*W + (x)) * 3];\
		dst[((y)*W + (x)) * 3 + 1] = src[((y)*W + (x)) * 3 + 1];\
		dst[((y)*W + (x)) * 3 + 2] = src[((y)*W + (x)) * 3 + 2]; }

#define IN_BOUNDS(x, y) (x > 0 && x < W && y > 0 && y < H)
#define gpuErrorCheck(a) { gpuAssert((a), __FILE__, __LINE__); }

#define ERRLOG(x) fprintf(stderr, "%s\n", x)
#define LOG(x) fprintf(stdout, "%s\n", x)

void inline gpuAssert(cudaError_t code, const char* file, int line, bool abort);
__host__ __device__ __declspec(noalias) unsigned int inline color_to_type(unsigned char r, unsigned char g, unsigned char b);
__device__ __declspec(noalias) int inline type_to_color(unsigned int t);
__global__ void kernel(unsigned char* pixels, unsigned char* buffer_pixels);
__global__ void kernel_update_pixels_buffer(unsigned char* pixels, unsigned char* buffer_pixels);
errno_t Init();
__declspec(noalias) void DrawCircle(float cx, float cy, float r, int num_segments);
void Render(unsigned char* pixels, unsigned char* d_pixels, unsigned char* d_buffer_pixels);
void Start();


#endif
