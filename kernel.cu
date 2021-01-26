/***

	Simple Cellular Automata based sandbox simulation by Miki

***/

// TODO: fix water

#include <SDL.h>
#include <glew.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <stdbool.h>

#include "Pixelworld.h"

/*
	air:	0x00, 0x00, 0x00 (black)
	water:	0x00, 0x00, 0xff (blue)
	sand:	0xff, 0xff, 0x00 (yellow)
	metal:	0x77, 0x77, 0x77 (grey)
*/

void inline gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

static int RADIUS = 1;
static int MOUSEX = 0, MOUSEY = 0;

static SDL_Window* w;
static SDL_GLContext glctx;
static bool quit = false;

__host__ __device__ __declspec(noalias) unsigned int inline color_to_type(unsigned char r, unsigned char g, unsigned char b) {
	//never return undefined
	return (r == 0xFF && g == 0xFF && b == 0x00) * SAND + (r == 0x00 && g == 0x00 && b == 0xFF) * WATER
		+ (r == 0x00 && g == 0x00 && b == 0x00) * AIR + (r == 0x77 && g == 0x77 && b == 0x77) * METAL;
}

__device__ __declspec(noalias) int inline type_to_color(unsigned int t) {
	//returns undefined
	return (t == 1) * 0x000000 + (t == 2) * 0xFFFF00 + (t == 3) * 0x0000FF + (t == 4) * 0x777777 + (t == 0) * 0x000000;
}

__global__ void kernel(unsigned char* pixels, unsigned char* buffer_pixels) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (idx > W*H)
		return;

	int x = idx % W;
	int y = (int)(idx / W);

	unsigned char r, g, b;
	GETPIXEL(pixels, r, g, b, x, y);

	unsigned int type = color_to_type(r, g, b);
	
	// sand blocks fall
	if (type == SAND && IN_BOUNDS(x, y)) {

		// 0th is bottom left, 1st is bottom, 2nd is bottom right
		char nbs[3] = {0, 0, 0};

		// test pixels below
		for (int i = -1; i <= 1; i++) {
			unsigned char n_r, n_g, n_b;
			GETPIXEL(pixels, n_r, n_g, n_b, x + i, y - 1);

			unsigned int n_type = color_to_type(n_r, n_g, n_b);

			if (n_type == SAND || n_type == METAL)
				nbs[i + 1] = 1;
		}

		// bottom is free => flow to bottom left => sand flows down
		if (!nbs[1]) {

			SETPIXEL(buffer_pixels, x, y - 1, SAND);
			SETPIXEL(buffer_pixels, x, y, AIR);
		}
		// bottom is occupied, but left is empty => sand flows down left
		else if (nbs[1] && !nbs[0]) {

			SETPIXEL(buffer_pixels, x - 1, y - 1, SAND);
			SETPIXEL(buffer_pixels, x, y, AIR);
		} 
		// only bottom right is free => flow down right
		else if (nbs[1] && nbs[0] && !nbs[2]) {

			SETPIXEL(buffer_pixels, x + 1, y - 1, SAND);
			SETPIXEL(buffer_pixels, x, y, AIR);
		} 
		// none of bottom pixels are free => stay put
		else if (nbs[1] && nbs[0] && nbs[2]) {

			SETPIXEL(buffer_pixels, x, y, SAND);
		}
	}
	// for metal blocks, they always stay put
	else if (type == METAL && IN_BOUNDS(x, y)) {

		SETPIXEL(buffer_pixels, x, y, METAL);
	}
	// water behaves just like sand, except it spreads horizontally as well
	else if (type == WATER && IN_BOUNDS(x, y)) {

		// 0th: bottom left, 1st: bottom, 2nd: bottom right, 3rd: left, 4th: right
		char nbs[5] = { 0, 0, 0, 0, 0 };

		// test pixels below
		for (int i = -1; i <= 1; i++) {
			unsigned char n_r, n_g, n_b;
			GETPIXEL(pixels, n_r, n_g, n_b, x + i, y - 1);

			unsigned int n_type = color_to_type(n_r, n_g, n_b);

			if (n_type == METAL || n_type == WATER)
				nbs[i + 1] = 1;
		}

		unsigned char r_r, r_g, r_b;
		GETPIXEL(pixels, r_r, r_g, r_b, x + 1, y);

		unsigned int r_type = color_to_type(r_r, r_g, r_b);
		if (r_type == METAL || r_type == WATER)
			nbs[4] = 1;

		unsigned char l_r, l_g, l_b;
		GETPIXEL(pixels, l_r, l_g, l_b, x - 1, y);

		unsigned int l_type = color_to_type(l_r, l_g, l_b);
		if (l_type == METAL || l_type == WATER)
			nbs[3] = 1;

		if (!nbs[1]) {

			SETPIXEL(buffer_pixels, x, y - 1, WATER);
			SETPIXEL(buffer_pixels, x, y, AIR);
		}
		// bottom is occupied, but left is empty => water flows down left
		else if (nbs[1] && !nbs[0]) {

			SETPIXEL(buffer_pixels, x - 1, y - 1, WATER);
			SETPIXEL(buffer_pixels, x, y, AIR);
		}
		// only bottom right is free => flow down right
		else if (nbs[1] && nbs[0] && !nbs[2]) {

			SETPIXEL(buffer_pixels, x + 1, y - 1, WATER);
			SETPIXEL(buffer_pixels, x, y, AIR);
		}
		// no bottom pixels available
		else if (nbs[1] && nbs[0] && nbs[2]) {
			// flow left
			if (!nbs[3]) {
				SETPIXEL(buffer_pixels, x - 1, y, WATER);
			
			}
			if (!nbs[4]) {
				SETPIXEL(buffer_pixels, x + 1, y, WATER);
			}
			SETPIXEL(buffer_pixels, x, y, AIR);

			if (nbs[3] && nbs[4]) {
				SETPIXEL(buffer_pixels, x, y, WATER);
			}

		}
		
	}
}

__global__ void kernel_update_pixels_buffer(unsigned char* pixels, unsigned char* buffer_pixels) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx > W*H)
		return;

	int x = idx % W;
	int y = (int)(idx / W);

	UPDATEPIXEL(pixels, buffer_pixels, x, y);

	if (IN_BOUNDS(x, y)) {

		SETPIXEL(buffer_pixels, x, y, AIR);
	}
}

errno_t Init() {
	if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
		return -1;

	w = SDL_CreateWindow("N-body simulation with CUDA", SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, W, H, SDL_WINDOW_OPENGL);
	if (w == NULL)
		return -2;

	glctx = SDL_GL_CreateContext(w);
	if (glctx == NULL)
		return -3;

	GLenum err = glewInit();
	if (err != GLEW_OK)
		return -4;

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	glClearColor(0, 0, 0, 1.0);


	return 0;
}

__declspec(noalias) void DrawCircle(float cx, float cy, float r, int num_segments) {
	glBegin(GL_LINE_LOOP);
	for (int ii = 0; ii < num_segments; ii++) {
		float theta = 2.0f * 3.1415926f * float(ii) / float(num_segments);
		float x = r * cosf(theta);
		float y = r * sinf(theta);
		glVertex2f(x + cx, y + cy);
	}
	glEnd();
}

void Render(unsigned char* pixels, unsigned char* d_pixels, unsigned char* d_buffer_pixels) {
	
	gpuErrorCheck(cudaMemcpy(d_pixels, pixels, W*H*BYTES_PER_COLOR * sizeof(char), cudaMemcpyHostToDevice));
	
	kernel <<<N_BLOCKS, N_THREADS>>> (d_pixels, d_buffer_pixels);
	gpuErrorCheck(cudaPeekAtLastError());
	kernel_update_pixels_buffer <<<N_BLOCKS, N_THREADS>>> (d_pixels, d_buffer_pixels);
	gpuErrorCheck(cudaPeekAtLastError());

	cudaDeviceSynchronize();

	gpuErrorCheck(cudaMemcpy(pixels, d_pixels, W*H * BYTES_PER_COLOR * sizeof(char), cudaMemcpyDeviceToHost));

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDrawPixels(W, H, GL_RGB, GL_UNSIGNED_BYTE, pixels);

	// render cursor halo
	DrawCircle((float)((float)(MOUSEX)/W*2 - 1.0), (float)(1.0 - (float)(MOUSEY)/H*2), (float)(RADIUS/500.0), 1000);

	SDL_GL_SwapWindow(w);
}

void Start() {

	unsigned char* pixels = (unsigned char*)malloc(BYTES_PER_COLOR*W*H * sizeof(char));
	unsigned char* d_pixels;
	unsigned char* d_buffer_pixels;

	gpuErrorCheck(cudaMalloc((void**)&d_pixels, BYTES_PER_COLOR*W*H * sizeof(char)));
	gpuErrorCheck(cudaMalloc((void**)&d_buffer_pixels, BYTES_PER_COLOR*W*H * sizeof(char)));

	// initialize pixels array
	memset(pixels, 0x00, W*H*BYTES_PER_COLOR);

	bool leftbuttondown = false;
	bool rightbuttondown = false;
	bool deletekey = false;
	bool waterkey = false;

	while (!quit) {
		SDL_Event e;
		while (SDL_PollEvent(&e) != NULL) {
			
			// get keyboard state
			const Uint8 *keys = SDL_GetKeyboardState(NULL);

			if (keys[SDL_SCANCODE_D]) {
				deletekey = true;
			}
			else {
				deletekey = false;
			}
			if (keys[SDL_SCANCODE_W]) {
				waterkey = true;
			}
			else {
				waterkey = false;
			}

			switch (e.type) {
			case SDL_QUIT:
				quit = true;
				break;
			case SDL_MOUSEWHEEL:
				if (e.wheel.y > 0) {
					RADIUS++;
				}
				else if (e.wheel.y < 0) {
					if(RADIUS>0) 
						RADIUS--;
				}
				break;
			case SDL_MOUSEBUTTONDOWN:
				if (e.button.button == SDL_BUTTON_LEFT && !leftbuttondown)
					leftbuttondown = true;
				if (e.button.button == SDL_BUTTON_RIGHT && !rightbuttondown)
					rightbuttondown = true;
				break;
			case SDL_MOUSEBUTTONUP:
				if (e.button.button == SDL_BUTTON_LEFT && leftbuttondown)
					leftbuttondown = false;
				if (e.button.button == SDL_BUTTON_RIGHT && rightbuttondown)
					rightbuttondown = false;
				break;
			case SDL_MOUSEMOTION:
				MOUSEX = e.motion.x;
				MOUSEY = e.motion.y;
				if (IN_BOUNDS(MOUSEX, MOUSEY)) {
					if (leftbuttondown) {
						int mx = e.motion.x;
						int my = e.motion.y;
						for (int i = 0; i < W; i++) {
							for (int j = 0; j < H; j++) {
								if ((i - mx)*(i - mx) + (j - my)*(j - my) <= RADIUS * RADIUS) {
									//draw only to empty pixels
									if (pixels[((H - j)*W + i) * 3] == 0x00 &&
										pixels[((H - j)*W + i) * 3 + 1] == 0x00 &&
										pixels[((H - j)*W + i) * 3 + 2] == 0x00) {

										pixels[((H - j)*W + i) * 3] = 0xFF;
										pixels[((H - j)*W + i) * 3 + 1] = 0xFF;
										pixels[((H - j)*W + i) * 3 + 2] = 0x00;
									}
								}
							}
						}
						//break;
					}
					else if (rightbuttondown) {
						int mx = e.motion.x;
						int my = e.motion.y;
						for (int i = 0; i < W; i++) {
							for (int j = 0; j < H; j++) {
								if ((i - mx)*(i - mx) + (j - my)*(j - my) <= RADIUS * RADIUS) {
									//draw only to empty pixels
									if (pixels[((H - j)*W + i) * 3] == 0x00 &&
										pixels[((H - j)*W + i) * 3 + 1] == 0x00 &&
										pixels[((H - j)*W + i) * 3 + 2] == 0x00) {

										pixels[((H - j)*W + i) * 3] = 0x77;
										pixels[((H - j)*W + i) * 3 + 1] = 0x77;
										pixels[((H - j)*W + i) * 3 + 2] = 0x77;
									}
								}
							}
						}
						//break;
					}
					else if (deletekey) {
						int mx = e.motion.x;
						int my = e.motion.y;
						for (int i = 0; i < W; i++) {
							for (int j = 0; j < H; j++) {
								if ((i - mx)*(i - mx) + (j - my)*(j - my) <= RADIUS * RADIUS) {
									//erase only non-air pixels
									unsigned int type = color_to_type(pixels[((H - j)*W + i) * 3],
										pixels[((H - j)*W + i) * 3 + 1], pixels[((H - j)*W + i) * 3 + 2]);

									if (type == METAL || type == SAND || type == WATER) {
										pixels[((H - j)*W + i) * 3] = 0x00;
										pixels[((H - j)*W + i) * 3 + 1] = 0x00;
										pixels[((H - j)*W + i) * 3 + 2] = 0x00;
									}
								}
							}
						}
						//break;
					}
					else if (waterkey) {
						int mx = e.motion.x;
						int my = e.motion.y;
						for (int i = 0; i < W; i++) {
							for (int j = 0; j < H; j++) {
								if ((i - mx)*(i - mx) + (j - my)*(j - my) <= RADIUS * RADIUS) {
									//draw only to empty pixels
									if (pixels[((H - j)*W + i) * 3] == 0x00 &&
										pixels[((H - j)*W + i) * 3 + 1] == 0x00 &&
										pixels[((H - j)*W + i) * 3 + 2] == 0x00) {

										pixels[((H - j)*W + i) * 3] = 0x00;
										pixels[((H - j)*W + i) * 3 + 1] = 0x00;
										pixels[((H - j)*W + i) * 3 + 2] = 0xFF;
									}
								}
							}
						}
						//break;
					}

				}
			default: {}
			}
		}

		Render(pixels, d_pixels, d_buffer_pixels);
	}

	SDL_GL_DeleteContext(glctx);
	SDL_DestroyWindow(w);
	SDL_Quit();

	cudaFree(d_pixels);
	cudaFree(d_buffer_pixels);
	free(pixels);

	pixels = NULL;
	d_pixels = NULL;
	d_buffer_pixels = NULL;
}

int main(int argc, char** argv) {

	errno_t err = Init();
	if (err < 0) {
		ERRLOG("[-] Error in Init");
		exit(-1);
	}

	LOG("[+] Starting");
	Start();
	LOG("[-] Quitting");

	return 0;
}
