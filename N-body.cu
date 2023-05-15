#include <glew.h>
#include <freeglut.h>

#include <cudaDefs.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <helper_math.h>

#include <benchmark.h>
#include <cuda_gl_interop.h>

#include <math_functions.h>

using std::cout;
using std::endl;

const int window_width = 1600;
const int window_height = 900;

GLuint vbo;
GLuint texID;
struct cudaGraphicsResource* cuda_vbo_resource;

using gpubenchmark::print_time;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

constexpr unsigned int numParticles = 16384;

constexpr unsigned int TPB = 128;

__constant__ double G = 6.6743e-7f;		//6.6743e-11f
__constant__ double Softenning = 1e-5f;	//1e-5f
__constant__ double Dt = 0.1f;		//0.01f

int previousTime = 0;
int frameCount = 0;

struct Particle {
	float4 position;
};
struct PhysicsOperations
{
	float4 velocity;
	float4 force;
};

Particle* particles;
PhysicsOperations* physicsOps;


// Implementation of parallel n-body simulation
__global__ void nBodySimulation(Particle* particles, PhysicsOperations* physicsOps, const int numParticles)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numParticles)
		return;

	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 pos1 = particles[i].position;
	double m1 = pos1.w;

	for (int j = 0; j < numParticles; j++) {
		if (j == i)
			continue;

		float4 pos2 = particles[j].position;
		float4 dist = pos2 - pos1;
		double r = length(dist);
		double m2 = pos2.w;
		double f = (G * (m1 * m2)) / ((r * r) + (Softenning * Softenning));
		force += f * normalize(dist);
	}

	float4 accel = force / m1;
	physicsOps[i].velocity += accel * Dt;
	particles[i].position += physicsOps[i].velocity * Dt;

}

//Implementation of parallel n - body simulation using shared memory
__global__ void nBodySimulationSM(Particle* particles, PhysicsOperations* physicsOps, const int numParticles)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;


	extern __shared__ float4 sharedPos[];

	if (i < numParticles)
	{
		float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		float4 pos1 = particles[i].position;
		double m1 = pos1.w;

		for (int blockOffset = 0; blockOffset < numParticles; blockOffset += blockDim.x)
		{
			int sharedIdx = blockOffset + threadIdx.x;

			sharedPos[threadIdx.x] = (sharedIdx < numParticles) ? particles[sharedIdx].position : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			//sharedPos[threadIdx.x] = (sharedIdx < numParticles) * particles[sharedIdx].position + (sharedIdx >= numParticles) * make_float4(0.0f, 0.0f, 0.0f, 0.0f);

			__syncthreads();

			#pragma unroll
			for (int j = 0; j < blockDim.x; j++)
			{
				if (blockOffset + j == i || blockOffset + j >= numParticles)
					continue;

				float4 pos2 = sharedPos[j];
				float4 dist = pos2 - pos1;
				double r = length(dist);
				double m2 = pos2.w;
				double f = (G * (m1 * m2)) / ((r * r) + (Softenning * Softenning)); //f = G * (m1 * m2) / r^2 + softenning^2
				force += f * normalize(dist);
			}

			__syncthreads();
		}

		float4 accel = force / m1;
		physicsOps[i].velocity += accel * Dt; //v = u +at
		particles[i].position += physicsOps[i].velocity * Dt; 
	}
}


void initializeOpenGLBuffers() 
{
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, numParticles * sizeof(float4), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
}

void initGL(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("N-Body Simulation");

	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0")) {
		cout << "OpenGL 2.0 not supported!" << endl;
		exit(EXIT_FAILURE);
	}

	glViewport(0, 0, window_width, window_height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-20.0f, 20.0f, -20.0f, 20.0f, -20.0f, 20.0f); 
	gluPerspective(60.0, (double)window_width / (double)window_height, 0.0001, 100.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_DEPTH_TEST);

	glClearColor(0.0, 0.0, 0.0, 0.0);

	initializeOpenGLBuffers();

	glEnable(GL_POINT_SPRITE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	
	glGenTextures(1, &texID);
	glBindTexture(GL_TEXTURE_2D, texID);
	unsigned char data[] = { 255, 255, 255, 255 };
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

}

// Basic render function
void renderA()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	float4* positions = (float4*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	for (int i = 0; i < numParticles; ++i)
	{
		positions[i] = particles[i].position;
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);


	glColor3f(1.0, 1.0, 0.0);
	glPointSize(2.0f);

	glDrawArrays(GL_POINTS, 0, numParticles);
	
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glutSwapBuffers();

	int currentTime = glutGet(GLUT_ELAPSED_TIME);
	frameCount++;
	if (currentTime - previousTime >= 1000)
	{
		float fps = frameCount / ((currentTime - previousTime) / 1000.0f);
		printf("FPS: %.2f\n", fps);
		previousTime = currentTime;
		frameCount = 0;
	}
}

//render with textures
void render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	float4* positions = (float4*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	for (int i = 0; i < numParticles; ++i)
	{
		positions[i] = particles[i].position;
		
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texID);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

	glPointSize(2.0f);
	glBegin(GL_POINTS);


	for (int i = 0; i < numParticles; ++i)
	{
		
		glPointSize(2.0f + positions[i].x * 10); // Not working without shaders
		//glColor3f(1.0f, 1.0f, 0.0f);
		glColor3f(abs(positions[i].x), abs(positions[i].y), 1.0f); 
		glTexCoord2f(0.0f, 0.0f);
		glVertex4f(positions[i].x, positions[i].y, positions[i].z, positions[i].w);
	}
	
	glEnd();


	glDisable(GL_TEXTURE_2D);
	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glutSwapBuffers();

	int currentTime = glutGet(GLUT_ELAPSED_TIME);
	frameCount++;
	if (currentTime - previousTime >= 1000)
	{
		float fps = frameCount / ((currentTime - previousTime) / 1000.0f);
		printf("FPS: %.2f\n", fps);
		previousTime = currentTime;
		frameCount = 0;
	}
}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);
	initGL(argc, argv);
	
	dim3 dimBlock = { TPB, 1, 1 };
	dim3 dimGrid = { getNumberOfParts(numParticles, TPB), 1, 1 };

	checkCudaErrors(cudaMallocManaged(&particles, numParticles * sizeof(Particle)));
	
	checkCudaErrors(cudaMallocManaged(&physicsOps, numParticles * sizeof(PhysicsOperations)));


	srand(static_cast<unsigned int>(time(nullptr)));

	for (int i = 0; i < numParticles; i++)
	{
		float x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f - 5.0f;
		float y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
		float z = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;

		float w = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 10.0f; //0.5f Blackhole like, 0.1f default

		particles[i].position = make_float4(x, y, z, w);

		physicsOps[i].force = make_float4(0, 0, 0, 0);
		physicsOps[i].velocity = make_float4(0, 0, 0, 0);
	}

	//glutDisplayFunc(render);

	for (int step = 0; step < 10000; step++)
	{
		//nBodySimulation << <dimGrid, dimBlock>> > (particles, physicsOps, numParticles);

		nBodySimulationSM << <dimGrid, dimBlock, TPB * sizeof(float4) >> > (particles, physicsOps, numParticles);
		cudaDeviceSynchronize();	
		render();
		
	}

	checkError();

	
	glutMainLoop();


	return 0;
}
