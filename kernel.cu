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

/*
	air:	0x00, 0x00, 0x00 (black)
	water:	0x00, 0x00, 0xff (blue)
	sand:	0xff, 0xff, 0x00 (yellow)
	metal:	0x77, 0x77, 0x77 (grey)
*/

#define W 1024
#define H 1024
#define BYTES_PER_COLOR 3

#define UNDEFINED 0
#define AIR 1
#define SAND 2
#define WATER 3
#define METAL 4

#define IN_BOUNDS(x, y) (x > 0 && x < W && y > 0 && y < H)

static int RADIUS = 1;
static int MOUSEX = 0, MOUSEY = 0;

static SDL_Window* w;
static SDL_GLContext glctx;
static bool quit = false;

__host__ __device__ unsigned int inline color_to_type(unsigned char r, unsigned char g, unsigned char b) {
	if (r == 0xFF && g == 0xFF && b == 0x00) {
		//sand
		return 2;
	}
	else if (r == 0x00 && g == 0x00 && b == 0xFF) {
		//water
		return 3;
	}
	else if (r == 0x00 && g == 0x00 && b == 0x00) {
		//air
		return 1;
	}
	else if (r == 0x77 && g == 0x77 && b == 0x77) {
		//metal
		return 4;
	}
	else {
		//undefined
		return 0;
	}
}

__device__ int inline type_to_color(unsigned int t) {
	if (t == 1) {
		//air
		return 0x000000;
	}
	else if(t == 2) {
		//sand
		return 0xFFFF00;
	}
	else if (t == 3) {
		//water
		return 0x0000FF;
	}
	else if (t == 4) {
		//metal
		return 0x777777;
	}
	else {
		//unhandled undefined
		return 0x000000;
	}
}

__global__ void kernel(unsigned char* pixels, unsigned char* buffer_pixels) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (idx > W*H)
		return;

	int x = idx % W;
	int y = (int)(idx / W);
	
	// color of current pixel
	unsigned char r = pixels[(y*W + x)*3];
	unsigned char g = pixels[(y*W + x) * 3 + 1];
	unsigned char b = pixels[(y*W + x) * 3 + 2];

	unsigned int type = color_to_type(r, g, b);
	
	// sand blocks fall
	if (type == SAND && IN_BOUNDS(x, y)) {

		// 0th is bottom left, 1st is bottom, 2nd is bottom right
		char nbs[3] = {0, 0, 0};

		// test pixels below
		for (int i = -1; i <= 1; i++) {
			unsigned char n_r = pixels[((y - 1)*W + (x + i)) * 3];
			unsigned char n_g = pixels[((y - 1)*W + (x + i)) * 3 + 1];
			unsigned char n_b = pixels[((y - 1)*W + (x + i)) * 3 + 2];

			unsigned int n_type = color_to_type(n_r, n_g, n_b);

			if (n_type == SAND || n_type == METAL)
				nbs[i + 1] = 1;
		}

		// bottom is free => flow to bottom left => sand flows down
		if (!nbs[1]) {
			buffer_pixels[((y - 1)*W + (x)) * 3] = 0xFF;
			buffer_pixels[((y - 1)*W + (x)) * 3 + 1] = 0xFF;
			buffer_pixels[((y - 1)*W + (x)) * 3 + 2] = 0x00;

			buffer_pixels[((y)*W + (x)) * 3] = 0x00;
			buffer_pixels[((y)*W + (x)) * 3 + 1] = 0x00;
			buffer_pixels[((y)*W + (x)) * 3 + 2] = 0x00;
		} 
		// bottom is occupied, but left is empty => sand flows down left
		else if (nbs[1] && !nbs[0]) {
			buffer_pixels[((y - 1)*W + (x - 1)) * 3] = 0xFF;
			buffer_pixels[((y - 1)*W + (x - 1)) * 3 + 1] = 0xFF;
			buffer_pixels[((y - 1)*W + (x - 1)) * 3 + 2] = 0x00;

			buffer_pixels[((y)*W + (x)) * 3] = 0x00;
			buffer_pixels[((y)*W + (x)) * 3 + 1] = 0x00;
			buffer_pixels[((y)*W + (x)) * 3 + 2] = 0x00;
		} 
		// only bottom right is free => flow down right
		else if (nbs[1] && nbs[0] && !nbs[2]) {
			buffer_pixels[((y - 1)*W + (x + 1)) * 3] = 0xFF;
			buffer_pixels[((y - 1)*W + (x + 1)) * 3 + 1] = 0xFF;
			buffer_pixels[((y - 1)*W + (x + 1)) * 3 + 2] = 0x00;

			buffer_pixels[((y)*W + (x)) * 3] = 0x00;
			buffer_pixels[((y)*W + (x)) * 3 + 1] = 0x00;
			buffer_pixels[((y)*W + (x)) * 3 + 2] = 0x00;
		} 
		// none of bottom pixels are free => stay put
		else if (nbs[1] && nbs[0] && nbs[2]) {
			buffer_pixels[((y)*W + (x)) * 3] = 0xFF;
			buffer_pixels[((y)*W + (x)) * 3 + 1] = 0xFF;
			buffer_pixels[((y)*W + (x)) * 3 + 2] = 0x00;
		}
	}
	// for metal blocks, they always stay put
	else if (type == METAL && IN_BOUNDS(x, y)) {
		buffer_pixels[((y)*W + (x)) * 3] = 0x77;
		buffer_pixels[((y)*W + (x)) * 3 + 1] = 0x77;
		buffer_pixels[((y)*W + (x)) * 3 + 2] = 0x77;
	}
	// water behaves just like sand, except it spreads horizontally
	else if (type == WATER && IN_BOUNDS(x, y)) {

		// 0th: bottom left, 1st: bottom, 2nd: bottom right, 3rd: left, 4th: right
		char nbs[5] = { 0, 0, 0, 0, 0 };

		// test pixels below
		for (int i = -1; i <= 1; i++) {
			unsigned char n_r = pixels[((y - 1)*W + (x + i)) * 3];
			unsigned char n_g = pixels[((y - 1)*W + (x + i)) * 3 + 1];
			unsigned char n_b = pixels[((y - 1)*W + (x + i)) * 3 + 2];

			unsigned int n_type = color_to_type(n_r, n_g, n_b);

			if (n_type == SAND || n_type == METAL || n_type == WATER)
				nbs[i + 1] = 1;
		}

		// test left and right pixels
		unsigned char r_r = pixels[((y)*W + (x + 1)) * 3];
		unsigned char r_g = pixels[((y)*W + (x + 1)) * 3 + 1];
		unsigned char r_b = pixels[((y)*W + (x + 1)) * 3 + 2];

		unsigned int r_type = color_to_type(r_r, r_g, r_b);
		if (r_type == SAND || r_type == METAL || r_type == WATER)
			nbs[4] = 1;

		unsigned char l_r = pixels[((y)*W + (x - 1)) * 3];
		unsigned char l_g = pixels[((y)*W + (x - 1)) * 3 + 1];
		unsigned char l_b = pixels[((y)*W + (x - 1)) * 3 + 2];

		unsigned int l_type = color_to_type(l_r, l_g, l_b);
		if (l_type == SAND || l_type == METAL || l_type == WATER)
			nbs[3] = 1;


		if (!nbs[1]) {
			buffer_pixels[((y - 1)*W + (x)) * 3] = 0x00;
			buffer_pixels[((y - 1)*W + (x)) * 3 + 1] = 0x00;
			buffer_pixels[((y - 1)*W + (x)) * 3 + 2] = 0xFF;

			buffer_pixels[((y)*W + (x)) * 3] = 0x00;
			buffer_pixels[((y)*W + (x)) * 3 + 1] = 0x00;
			buffer_pixels[((y)*W + (x)) * 3 + 2] = 0x00;
		}
		// bottom is occupied, but left is empty => water flows down left
		else if (nbs[1] && !nbs[0]) {
			buffer_pixels[((y - 1)*W + (x - 1)) * 3] = 0x00;
			buffer_pixels[((y - 1)*W + (x - 1)) * 3 + 1] = 0x00;
			buffer_pixels[((y - 1)*W + (x - 1)) * 3 + 2] = 0xFF;

			buffer_pixels[((y)*W + (x)) * 3] = 0x00;
			buffer_pixels[((y)*W + (x)) * 3 + 1] = 0x00;
			buffer_pixels[((y)*W + (x)) * 3 + 2] = 0x00;
		}
		// only bottom right is free => flow down right
		else if (nbs[1] && nbs[0] && !nbs[2]) {
			buffer_pixels[((y - 1)*W + (x + 1)) * 3] = 0x00;
			buffer_pixels[((y - 1)*W + (x + 1)) * 3 + 1] = 0x00;
			buffer_pixels[((y - 1)*W + (x + 1)) * 3 + 2] = 0xFF;

			buffer_pixels[((y)*W + (x)) * 3] = 0x00;
			buffer_pixels[((y)*W + (x)) * 3 + 1] = 0x00;
			buffer_pixels[((y)*W + (x)) * 3 + 2] = 0x00;
		}
		// none of bottom pixels are free => look for sides
		else if (nbs[1] && nbs[0] && nbs[2]) {
			// if left is empty
			if (!nbs[3]) {
				buffer_pixels[((y)*W + (x - 1)) * 3] = 0x00;
				buffer_pixels[((y)*W + (x - 1)) * 3 + 1] = 0x00;
				buffer_pixels[((y)*W + (x - 1)) * 3 + 2] = 0xFF;

				buffer_pixels[((y)*W + (x)) * 3] = 0x00;
				buffer_pixels[((y)*W + (x)) * 3 + 1] = 0x00;
				buffer_pixels[((y)*W + (x)) * 3 + 2] = 0x00;

			// if left is occupied, but right is empty
			}
			else if (nbs[3] && !nbs[4]) {
				buffer_pixels[((y)*W + (x + 1)) * 3] = 0x00;
				buffer_pixels[((y)*W + (x + 1)) * 3 + 1] = 0x00;
				buffer_pixels[((y)*W + (x + 1)) * 3 + 2] = 0xFF;

				buffer_pixels[((y)*W + (x)) * 3] = 0x00;
				buffer_pixels[((y)*W + (x)) * 3 + 1] = 0x00;
				buffer_pixels[((y)*W + (x)) * 3 + 2] = 0x00;
			} 
			// both direction
			else if (!nbs[3] && !nbs[4]) {
				buffer_pixels[((y)*W + (x - 1)) * 3] = 0x00;
				buffer_pixels[((y)*W + (x - 1)) * 3 + 1] = 0x00;
				buffer_pixels[((y)*W + (x - 1)) * 3 + 2] = 0xFF;

				buffer_pixels[((y)*W + (x + 1)) * 3] = 0x00;
				buffer_pixels[((y)*W + (x + 1)) * 3 + 1] = 0x00;
				buffer_pixels[((y)*W + (x + 1)) * 3 + 2] = 0xFF;

				buffer_pixels[((y)*W + (x)) * 3] = 0x00;
				buffer_pixels[((y)*W + (x)) * 3 + 1] = 0x00;
				buffer_pixels[((y)*W + (x)) * 3 + 2] = 0x00;
			}
			// neither => stay put
			else {
				buffer_pixels[((y)*W + (x)) * 3] = 0x00;
				buffer_pixels[((y)*W + (x)) * 3 + 1] = 0x00;
				buffer_pixels[((y)*W + (x)) * 3 + 2] = 0xFF;
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

	pixels[((y)*W + (x)) * 3] = buffer_pixels[((y)*W + (x)) * 3];
	pixels[((y)*W + (x)) * 3 + 1] = buffer_pixels[((y)*W + (x)) * 3 + 1];
	pixels[((y)*W + (x)) * 3 + 2] = buffer_pixels[((y)*W + (x)) * 3 + 2];

	if (IN_BOUNDS(x, y)) {
		buffer_pixels[((y)*W + (x)) * 3] = 0x00;
		buffer_pixels[((y)*W + (x)) * 3 + 1] = 0x00;
		buffer_pixels[((y)*W + (x)) * 3 + 2] = 0x00;
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

void DrawCircle(float cx, float cy, float r, int num_segments) {
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
	cudaMemcpy(d_pixels, pixels, W*H*BYTES_PER_COLOR * sizeof(char), cudaMemcpyHostToDevice);
	
	kernel <<<W, H>>> (d_pixels, d_buffer_pixels);
	kernel_update_pixels_buffer <<<W, H>>> (d_pixels, d_buffer_pixels);
	
	cudaDeviceSynchronize();

	cudaMemcpy(pixels, d_pixels, W*H * BYTES_PER_COLOR * sizeof(char), cudaMemcpyDeviceToHost);

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

	cudaMalloc((void**)&d_pixels, BYTES_PER_COLOR*W*H * sizeof(char));
	cudaMalloc((void**)&d_buffer_pixels, BYTES_PER_COLOR*W*H * sizeof(char));

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
									unsigned int type = color_to_type(pixels[((H-j)*W + i) * 3], 
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
		fprintf(stderr, "[-] Error: %d\n", err);
		exit(-1);
	}

	fprintf(stdout, "[+] Starting\n");
	Start();
	fprintf(stdout, "[-] Quitting\n");

	return 0;
}