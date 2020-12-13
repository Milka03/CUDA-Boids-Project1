// Emilia Wróblewska

// OpenGL
#define GLEW_STATIC
#include <GL\glew.h>
#include <GLFW\glfw3.h>
//#include <gl\gl.h>

// CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>

#define USE_GPU true

#define blockSize 128
#define ITERATIONS 10000
#define WIDTH 1840
#define HEIGHT 1024

#define boidsCount 10000
#define boidMaxSpeed 1.0f
#define rule1Distance 20.0f
#define rule2Distance 10.0f
#define rule3Distance 20.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define cursorDistance 200
#define cursorEscapeVelocity 1.0f

// Calcualte distance between two boids/length of velocity vector
__device__ float magnitude(float x, float y) {
	return (float)sqrt(x * x + y * y);
}

float magnitudeCPU(float x, float y) {
	return (float)sqrt(x * x + y * y);
}

// number of blocks
int numBlocks = (boidsCount + blockSize - 1) / blockSize;

// is cursor over window
bool cursorOverWindow = false;
double cursorX;
double cursorY;
bool Moving = true;
// host boid position
float* x;
float* y;
// device boid position
float* p_x;
float* p_y;
// old velocity
float* v_x1;
float* v_y1;
// new velocity
float* v_x2;
float* v_y2;


//---------- Initializing arrays, allocating shared memory -----------
void initializeBoids(float* vColors) 
{
	cudaError_t cudaStatus;
	cudaMallocManaged(&p_x, boidsCount * sizeof(float));
	cudaMallocManaged(&p_y, boidsCount * sizeof(float));
	cudaMallocManaged(&v_x1, boidsCount * sizeof(float));
	cudaMallocManaged(&v_y1, boidsCount * sizeof(float));
	cudaMallocManaged(&v_x2, boidsCount * sizeof(float));
	cudaMallocManaged(&v_y2, boidsCount * sizeof(float));

	float* tempX = (float*)malloc(boidsCount * sizeof(float));
	float* tempY = (float*)malloc(boidsCount * sizeof(float));
	int r, g, b;

	for (int i = 0; i < boidsCount; i++) 
	{
		x[i] = ((float)rand() / RAND_MAX) * WIDTH;
		y[i] = ((float)rand() / RAND_MAX) * HEIGHT;

		tempX[i] = (rand() % 2) + ((float)rand() / RAND_MAX) - 1.0;
		tempY[i] = (rand() % 2) + ((float)rand() / RAND_MAX) - 1.0;

		r = rand() % 2; g = rand() % 2;
		if (r == 0 && g == 0) b = 1; //avoid black color 
		else b = rand() % 2;
		vColors[9 * i + 0] = (float)r;
		vColors[9 * i + 1] = (float)g;
		vColors[9 * i + 2] = (float)b;
		vColors[9 * i + 3] = (float)r;
		vColors[9 * i + 4] = (float)g;
		vColors[9 * i + 5] = (float)b;
		vColors[9 * i + 6] = (float)r;
		vColors[9 * i + 7] = (float)g;
		vColors[9 * i + 8] = (float)b;
	}

	cudaStatus = cudaMemcpy(p_x, x, boidsCount * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(p_y, y, boidsCount * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(v_x1, tempX, boidsCount * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(v_y1, tempY, boidsCount * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(v_x2, tempX, boidsCount * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(v_y2, tempY, boidsCount * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)  fprintf(stderr, "cudaMemcpy failed!");
	
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

	free(tempX);
	free(tempY);
}

void initializeBoidsCPU(float* vColors) 
{
	v_x1 = (float*)malloc(boidsCount * sizeof(float));
	v_y1 = (float*)malloc(boidsCount * sizeof(float));
	v_x2 = (float*)malloc(boidsCount * sizeof(float));
	v_y2 = (float*)malloc(boidsCount * sizeof(float));

	float r, g, b;
	for (int i = 0; i < boidsCount; i++) 
	{
		x[i] = ((float)rand() / RAND_MAX) * WIDTH;
		y[i] = ((float)rand() / RAND_MAX) * HEIGHT;

		v_x1[i] = (rand() % 2) + ((float)rand() / RAND_MAX) - 1.0;
		v_x2[i] = v_x1[i];
		v_y1[i] = (rand() % 2) + ((float)rand() / RAND_MAX) - 1.0;
		v_y2[i] = v_y1[i];

		r = rand() % 2; g = rand() % 2;
		if (r == 0 && g == 0) b = 1; //avoid black color 
		else b = rand() % 2;
		vColors[9 * i + 0] = (float)r;
		vColors[9 * i + 1] = (float)g;
		vColors[9 * i + 2] = (float)b;
		vColors[9 * i + 3] = (float)r;
		vColors[9 * i + 4] = (float)g;
		vColors[9 * i + 5] = (float)b;
		vColors[9 * i + 6] = (float)r;
		vColors[9 * i + 7] = (float)g;
		vColors[9 * i + 8] = (float)b;
	}
}


//---------- Free memory -----------
void freeBoids() {
	cudaFree(p_x);
	cudaFree(p_y);
	cudaFree(v_x1);
	cudaFree(v_y1);
	cudaFree(v_x2);
	cudaFree(v_y2);
}

void freeBoidsCPU() {
	free(v_x1);
	free(v_x2);
	free(v_y1);
	free(v_y2);
}


//--------------- Calculate new velocity using 3 rules of flocking behaviour ----------------

__device__ void newVelocity(int index, float* pos_x, float* pos_y, float* vel_x1, float* vel_y1, float* vel_x2, float* vel_y2, bool isCursorOverWindow, float curX, float curY)
{
	float v1[2] = { 0.0f, 0.0f };
	float v2[2] = { 0.0f, 0.0f };
	float v3[2] = { 0.0f, 0.0f };

	float percieved_center_of_mass[2] = { 0.0f, 0.0f };
	float perceived_velocity[2] = { 0.0f, 0.0f };
	float separate_vector[2] = { 0.0f, 0.0f };

	int neighborCount1 = 0;
	int neighborCount3 = 0;
	float distance = 0.0f;

	for (int i = 0; i < boidsCount; i++)
	{
		if (i != index)
		{
			distance = magnitude(pos_x[i] - pos_x[index], pos_y[i] - pos_y[index]);
			//Rule 1: Cohesion
			if (distance < rule1Distance)
			{
				percieved_center_of_mass[0] += pos_x[i];
				percieved_center_of_mass[1] += pos_y[i];
				neighborCount1++;
			}
			//Rule 2: Separation
			if (distance < rule2Distance)
			{
				separate_vector[0] -= (pos_x[i] - pos_x[index]);
				separate_vector[1] -= (pos_y[i] - pos_y[index]);
			}
			//Rule 3: Alignment
			if (distance < rule3Distance)
			{
				perceived_velocity[0] += vel_x1[i];
				perceived_velocity[1] += vel_y1[i];
				neighborCount3++;
			}
		}
	}
	// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
	if (neighborCount1 != 0)
	{
		percieved_center_of_mass[0] /= neighborCount1;
		percieved_center_of_mass[1] /= neighborCount1;
		v1[0] = (percieved_center_of_mass[0] - pos_x[index]) * rule1Scale;
		v1[1] = (percieved_center_of_mass[1] - pos_y[index]) * rule1Scale;
	}
	// Rule 2: boids try to stay a distance d away from each other
	v2[0] = separate_vector[0] * rule2Scale;
	v2[1] = separate_vector[1] * rule2Scale;

	// Rule 3: boids try to match the speed of surrounding boids
	if (neighborCount3 != 0)
	{
		perceived_velocity[0] /= neighborCount3;
		perceived_velocity[1] /= neighborCount3;
		v3[0] = perceived_velocity[0] * rule3Scale;
		v3[1] = perceived_velocity[1] * rule3Scale;
	}

	float newVelX = vel_x1[index] + v1[0] + v2[0] + v3[0];
	float newVelY = vel_y1[index] + v1[1] + v2[1] + v3[1];

	//Fly towards cursor
	if (isCursorOverWindow) {
		distance = magnitude(curX - pos_x[index], curY - pos_y[index]);
		if (distance < cursorDistance) {
			newVelX += (curX - pos_x[index]); 
			newVelY += (curY - pos_y[index]);
		}
	}
	//Clamp the speed
	float length = magnitude(newVelX, newVelY);
	if (length > boidMaxSpeed) {
		newVelX = (newVelX / length) * boidMaxSpeed;
		newVelY = (newVelY / length) * boidMaxSpeed;
	}
	vel_x2[index] = newVelX;
	vel_y2[index] = newVelY;
}


__global__ void kernelUpdateVelocity(int N, float* pos_x, float* pos_y, float* vel_x1, float* vel_y1, float* vel_x2, float* vel_y2, bool isCursorOverWindow, float curX, float curY) 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N)
		return;

	newVelocity(index, pos_x, pos_y, vel_x1, vel_y1, vel_x2, vel_y2, isCursorOverWindow, curX, curY);
}


void newVelocityCPU(bool isCursorOverWindow, float curX, float curY) 
{
	for (int index = 0; index < boidsCount; index++) 
	{
		float v1[2] = { 0.0f, 0.0f };
		float v2[2] = { 0.0f, 0.0f };
		float v3[2] = { 0.0f, 0.0f };

		float percieved_center_of_mass[2] = { 0.0f, 0.0f };
		float perceived_velocity[2] = { 0.0f, 0.0f };
		float separate_vector[2] = { 0.0f, 0.0f };

		int neighborCount1 = 0;
		int neighborCount3 = 0;
		float distance = 0.0f;

		for (int i = 0; i < boidsCount; i++)
		{
			if (i != index)
			{
				distance = magnitudeCPU(x[i] - x[index], y[i] - y[index]);
				//Rule 1: Cohesion
				if (distance < rule1Distance)
				{
					percieved_center_of_mass[0] += x[i];
					percieved_center_of_mass[1] += y[i];
					neighborCount1++;
				}
				//Rule 2: Separation
				if (distance < rule2Distance)
				{
					separate_vector[0] -= (x[i] - x[index]);
					separate_vector[1] -= (y[i] - y[index]);
				}
				//Rule 3: Alignment
				if (distance < rule3Distance)
				{
					perceived_velocity[0] += v_x1[i];
					perceived_velocity[1] += v_y1[i];
					neighborCount3++;
				}
			}
		}
		// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
		if (neighborCount1 != 0)
		{
			percieved_center_of_mass[0] /= neighborCount1;
			percieved_center_of_mass[1] /= neighborCount1;
			v1[0] = (percieved_center_of_mass[0] - x[index]) * rule1Scale;
			v1[1] = (percieved_center_of_mass[1] - y[index]) * rule1Scale;
		}
		// Rule 2: boids try to stay a distance d away from each other
		v2[0] = separate_vector[0] * rule2Scale;
		v2[1] = separate_vector[1] * rule2Scale;

		// Rule 3: boids try to match the speed of surrounding boids
		if (neighborCount3 != 0)
		{
			perceived_velocity[0] /= neighborCount3;
			perceived_velocity[1] /= neighborCount3;
			v3[0] = perceived_velocity[0] * rule3Scale;
			v3[1] = perceived_velocity[1] * rule3Scale;
		}

		float newVelX = v_x1[index] + v1[0] + v2[0] + v3[0];
		float newVelY = v_y1[index] + v1[1] + v2[1] + v3[1];

		//Fly towards cursor
		if (isCursorOverWindow) {
			distance = magnitudeCPU(curX - x[index], curY - y[index]);
			if (distance < cursorDistance) {
				newVelX += (curX - x[index]); /// distance) * cursorEscapeVelocity;
				newVelY += (curY - y[index]);
			}
		}
		//Clamp the speed
		float length = magnitudeCPU(newVelX, newVelY);
		if (length > boidMaxSpeed) {
			newVelX = (newVelX / length) * boidMaxSpeed;
			newVelY = (newVelY / length) * boidMaxSpeed;
		}
		v_x2[index] = newVelX;
		v_y2[index] = newVelY;
	}
}


//---------------- Update position based on new velocity ------------------

__global__ void kernelUpdatePosition(int N, float* pos_x, float* pos_y, float* vel_x1, float* vel_y1) 
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N)
		return;

	pos_x[index] = pos_x[index] + vel_x1[index];
	pos_y[index] = pos_y[index] + vel_y1[index];

	if (pos_x[index] > (float)WIDTH)
		pos_x[index] = 0.0;
	if (pos_x[index] < 0.0)
		pos_x[index] = (float)WIDTH;

	if (pos_y[index] > (float)HEIGHT)
		pos_y[index] = 0.0;
	if (pos_y[index] < 0.0)
		pos_y[index] = (float)HEIGHT;
}

void updatePositionCPU() {
	for (int index = 0; index < boidsCount; index++) {
		x[index] = x[index] + v_x2[index];
		y[index] = y[index] + v_y2[index];

		if (x[index] > (float)WIDTH)
			x[index] = 0.0;
		if (x[index] < 0.0)
			x[index] = (float)WIDTH;

		if (y[index] > (float)HEIGHT)
			y[index] = 0.0;
		if (y[index] < 0.0)
			y[index] = (float)HEIGHT;
	}
}


//---------------- Calculating new boids positions in one glfw window loop iteration -----------------

void oneStepIteration(float* pos_x, float* pos_y, float* vel_x1, float* vel_y1, float* vel_x2, float* vel_y2) {
	cudaError_t cudaStatus;
	
	kernelUpdateVelocity <<<numBlocks, blockSize>>>((int)boidsCount, p_x, p_y, v_x1, v_y1, v_x2, v_y2, cursorOverWindow, (float)cursorX, (float)cursorY);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);

	kernelUpdatePosition <<<numBlocks, blockSize>>>((int)boidsCount, p_x, p_y, v_x2, v_y2);

	cudaStatus = cudaMemcpy(v_x1, v_x2, boidsCount * sizeof(float), cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed!");

	cudaStatus = cudaMemcpy(v_y1, v_y2, boidsCount * sizeof(float), cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaMemcpy failed!");

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
}

// function for one step iteration with CPU
void oneStepIterationCPU() {
	newVelocityCPU(cursorOverWindow, (float)cursorX, (float)cursorY);
	updatePositionCPU();
	memcpy(v_x1, v_x2, boidsCount * sizeof(float));
	memcpy(v_y1, v_y2, boidsCount * sizeof(float));
}


//---------------- Copying new positions of boids to one Vertex Array ----------------

__global__ void prepareVertices(int N, float* vertices_position, float* pos_X, float* pos_Y, float* vel_x, float* vel_y, int step)
{
	int index_0 = blockIdx.x * blockDim.x + threadIdx.x;
	int index = ((index_0) * 6) + step;
	if (index >= N)
		return;

	//We draw boids as eqiulateral triangles of height 5*sqrt(3) heading in direction pointed by point (pos_X[index_0], pos_Y[index_0])
	float s = sqrtf(3);
	float triangle_h = 5 * s;
	float vector_length = magnitude(vel_x[index_0], vel_y[index_0]);
	float h_x = pos_X[index_0] - (triangle_h * (vel_x[index_0] / vector_length));
	float h_y = pos_Y[index_0] - (triangle_h * (vel_y[index_0] / vector_length));

	if (step == 0) {  //when we fill x coordinates of all vertices
		vertices_position[index] = pos_X[index_0];
		vertices_position[index + 2] = h_x + ((pos_Y[index_0] - h_y)/s); 
		vertices_position[index + 4] = h_x + ((h_y - pos_Y[index_0])/s);
	}
	if (step == 1) { //when we fill y coordinates of all vertices
		vertices_position[index] = pos_Y[index_0];
		vertices_position[index + 2] = h_y + ((h_x - pos_X[index_0])/s);
		vertices_position[index + 4] = h_y + ((pos_X[index_0] - h_x)/s);
	}
}


void prepareBoidsToDraw(float* vertices_position) {
	cudaError_t cudaStatus;

	prepareVertices <<<numBlocks, blockSize >>> (6 * (int)boidsCount, vertices_position, p_x, p_y, v_x2, v_y2, 0); //for x's in VAO
	prepareVertices <<<numBlocks, blockSize >>> (6 * (int)boidsCount, vertices_position, p_x, p_y, v_x2, v_y2, 1); //for y's in VAO

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}
}

void prepareBoidsToDrawCPU(float* vertices_position)
{
	//We draw boids as eqiulateral triangles of height 5*sqrt(3) heading in direction pointed by point (x[i],y[i])
	float s = sqrtf(3);
	float triangle_h = 5 * s;
	float vector_length = 0; 
	float h_x = 0;
	float h_y = 0;

	for (int i = 0, j = 0; i < boidsCount; i++, j++) 
	{
		vector_length = magnitudeCPU(v_x2[i], v_y2[i]);
		h_x = x[i] - (triangle_h * (v_x2[i] / vector_length));
		h_y = y[i] - (triangle_h * (v_y2[i] / vector_length));

		vertices_position[j] = x[i]; 
		vertices_position[++j] = y[i]; 
		vertices_position[++j] = h_x + ((y[i] - h_y)/s) ;
		vertices_position[++j] = h_y + ((h_x - x[i])/s);
		vertices_position[++j] = h_x + ((h_y - y[i])/s);
		vertices_position[++j] = h_y + ((x[i] - h_x)/s);
	}
}


//---------------- Display FPS as window title / Window Callbacks ----------------

void displayFPS(GLFWwindow* window, double frameCount) {
	std::ostringstream ss;
	ss.precision(1);
	ss << std::fixed << frameCount;
	ss << " [fps] ";
	glfwSetWindowTitle(window, ss.str().c_str());
}

void CursorEnterCallback(GLFWwindow* window, int entered) {
	if (entered) {
		cursorOverWindow = true;
	}
	else {
		cursorOverWindow = false;
	}
}

void CursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
	cursorX = xpos;
	cursorY = ypos;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		bool tmp = !Moving;
		Moving = tmp;
	}
}


// -------------------------- MAIN FUNCTION ---------------------------

int main(int argc, char** argv) 
{
	srand(time(NULL));
	//Initialize boids positions and colors
	x = (float*)malloc(boidsCount * sizeof(float));
	y = (float*)malloc(boidsCount * sizeof(float));
	float* vertexColors = (float*)malloc(9 * boidsCount * sizeof(float));

	if (USE_GPU == true) initializeBoids(vertexColors);
	else initializeBoidsCPU(vertexColors);

	// OpenGL
	if (glfwInit() != GL_TRUE) {
		std::cerr << "Fail to initialize GLFW\n";
		return -1;
	}
	else printf("OPENGL OK\n\nPress [Space] to stop/resume boids movement.\n\n");

	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Project boids", nullptr, nullptr); // window pointer
	if (!window) std::cerr << "glfwCreateWindow error\n";
	glfwMakeContextCurrent(window);		// rendering
	glfwSetCursorEnterCallback(window, CursorEnterCallback);
	glfwSetKeyCallback(window, keyCallback);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cerr << "Fail to initialize GLEW\n";
		return -1;
	}
	glViewport(0, 0, WIDTH, HEIGHT);
	glOrtho(0.0f, WIDTH, HEIGHT, 0.0f, 0.0f, 1.0f);

	float* vertices_position; 
	if (USE_GPU == true) {
		cudaMallocManaged(&vertices_position, 6 * boidsCount * sizeof(float));
		prepareBoidsToDraw(vertices_position); 
	}
	else {
		vertices_position = (float*)malloc(6 * boidsCount * sizeof(float));
		prepareBoidsToDrawCPU(vertices_position); 
	}
	
	// FPS and loop time variables
	int iter = 0;
	int frameCount = 0;
	double fps = 0;
	double timebase = 0;
	clock_t start = clock();
	double previousTime = glfwGetTime();

	while (!glfwWindowShouldClose(window)) 
	{
		double currentTime = glfwGetTime();
		if (cursorOverWindow) glfwGetCursorPos(window, &cursorX, &cursorY);

		//DRAW
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_POINT_SMOOTH);
		glEnableClientState(GL_VERTEX_ARRAY); 
		glPointSize(2.0f);
		glVertexPointer(2, GL_FLOAT, 0, vertices_position); //link array with coordinates of vertices

		glEnableClientState(GL_COLOR_ARRAY);
		glColorPointer(3, GL_FLOAT, 0, vertexColors); //link array with colors of vertices

		glDrawArrays(GL_TRIANGLES, 0, 3 * boidsCount);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
		glDisable(GL_POINT_SMOOTH);

		// move boids
		if (Moving) {
			if (USE_GPU == true) {
				oneStepIteration(p_x, p_y, v_x1, v_y1, v_x2, v_y2);
				prepareBoidsToDraw(vertices_position);
			}
			else {
				oneStepIterationCPU();
				prepareBoidsToDrawCPU(vertices_position); 
			}
		}
		// FPS
		frameCount++;
		if (currentTime - timebase >= 1.0) {
			fps = frameCount / (currentTime - timebase);
			timebase = currentTime;
			frameCount = 0;
			displayFPS(window, fps);
		}
		//Avoid infite loop
		if (++iter >= ITERATIONS) glfwSetWindowShouldClose(window, GLFW_TRUE);

		glfwPollEvents();
		glfwSwapBuffers(window);
	}

	clock_t end = clock();
	float loopTime = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Time: %f\n", loopTime);

	glfwTerminate();
	if (USE_GPU == true) {
		freeBoids();
		cudaFree(vertices_position);
	}
	else {
		freeBoidsCPU();
		free(vertices_position);
	}
	free(x);
	free(y);
	free(vertexColors);
}