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

const int numParticles = 256;

const int numBlocks = 256;
const int blockSize = numParticles / numBlocks;

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

	const double G = 6.6743e-6; //e-11
	const double Softenning = 1e-5;
	const double Dt = 0.01;

	//printf("G | S | Dt (%f, %f, %f)\n", G, Softenning, Dt);

	if (i >= numParticles)
		return;

	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 pos1 = particles[i].position;
	double m1 = pos1.w;
	//printf("pos1 (%f, %f, %f, %f)\n", pos1.x, pos1.y, pos1.z, pos1.w);
	//printf("m1 (%f)\n", m1);

	for (int j = 0; j < numParticles; j++) {
		if (j == i)
			continue;

		float4 pos2 = particles[j].position;
		float4 dist = pos2 - pos1;
		double r = length(dist);
		double m2 = pos2.w;
		double f = (G * (m1 * m2)) / ((r * r) + (Softenning * Softenning));
		force += f * normalize(dist);

		//printf("pos2 (%f, %f, %f, %f)\n", pos2.x, pos2.y, pos2.z, pos2.w);
		//printf("dist (%f, %f, %f)\n", dist.x, dist.y, dist.z, dist.w);
		//printf("r (%f)\n", r);
		//printf("m2 (%f)\n", m2);
		//printf("f (%f)\n", f);
		//printf("Force (%f, %f, %f)\n", force.x, force.y, force.z);
		//printf("\n");
	}

	float4 accel = force / m1;
	physicsOps[i].velocity += accel * Dt;
	//printf("Velocity (%f, %f, %f)\n", physicsOps[i].velocity.x, physicsOps[i].velocity.y, physicsOps[i].velocity.z);
	particles[i].position += physicsOps[i].velocity * Dt;


}

//Implementation of parallel n - body simulation using shared memory
__global__ void nBodySimulationSM(Particle* particles, PhysicsOperations* physicsOps, const int numParticles)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	const double G = 6.6743e-8;
	const double Softenning = 1e-5;
	const double Dt = 0.01;

	if (i >= numParticles)
		return;

	__shared__ float4 sharedPos[numBlocks];
	float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	float4 pos1 = particles[i].position;
	double m1 = pos1.w;

	for (int blockOffset = 0; blockOffset < numParticles; blockOffset += blockDim.x) 
	{
		int sharedIdx = blockOffset + threadIdx.x;

		if (sharedIdx < numParticles) 
		{
			sharedPos[threadIdx.x] = particles[sharedIdx].position;
		}
		__syncthreads();

		for (int j = 0; j < blockDim.x; j++) 
		{
			if (blockOffset + j == i || blockOffset + j >= numParticles)
				continue;

			float4 pos2 = sharedPos[j];
			float4 dist = pos2 - pos1;
			double r = length(dist);
			double m2 = pos2.w;
			double f = (G * (m1 * m2)) / ((r * r) + (Softenning * Softenning));
			force += f * normalize(dist);
		}
		__syncthreads();
	}

	float4 accel = force / m1;
	physicsOps[i].velocity += accel * Dt;
	particles[i].position += physicsOps[i].velocity * Dt;
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
	glOrtho(-10.0f, 10.0f, -10.0f, 10.0f, -10.0f, 10.0f); //Old as the world itself but working
	//gluPerspective(60.0, (double)window_width / (double)window_height, 0.0001, 1000.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glEnable(GL_DEPTH_TEST);

	glClearColor(0.0, 0.0, 0.0, 0.0);



	initializeOpenGLBuffers();

	glEnable(GL_POINT_SPRITE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Load texture for point sprite
	
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

	// Render velocity vectors
	/*glColor3f(1.0, 0.0, 0.0);
	for (int i = 0; i < numParticles; ++i)
	{
		float4 pos = particles[i].position;
		float4 vel = physicsOps[i].velocity;
		float4 end = pos + vel;

		glBegin(GL_LINES);
		glVertex3f(pos.x, pos.y, pos.z);
		glVertex3f(end.x, end.y, end.z);
		glEnd();


		glBegin(GL_TRIANGLES);
		glVertex3f(end.x, end.y, end.z);
		glEnd();
	}*/

	glDisableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glutSwapBuffers();
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
	glDrawArrays(GL_POINTS, 0, numParticles);

	glDisable(GL_TEXTURE_2D);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
}

int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);
	initGL(argc, argv);
	
	

	checkCudaErrors(cudaMallocManaged(&particles, numParticles * sizeof(Particle)));
	
	checkCudaErrors(cudaMallocManaged(&physicsOps, numParticles * sizeof(PhysicsOperations)));


	srand(time(NULL));

	for (int i = 0; i < numParticles; i++)
	{
		float x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
		float y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
		float z = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;

		float w = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.1f + 0.1f; //0.5f Blackhole like, 0.1f default

		particles[i].position = make_float4(x, y, z, w);

		physicsOps[i].force = make_float4(0, 0, 0, 0);
		physicsOps[i].velocity = make_float4(0, 0, 0, 0);
	}

	
	glutDisplayFunc(render);

	for (int i = 0; i < numParticles; i++)
	{
		printf("Particle %d: (%f, %f, %f)\n", i, particles[i].position.x, particles[i].position.y, particles[i].position.z);
	}


	for (int step = 0; step < 10000; step++)
	{
		//nBodySimulation << <numBlocks, blockSize>> > (particles, physicsOps, numParticles);

		nBodySimulationSM << <numBlocks, blockSize >> > (particles, physicsOps, numParticles);
		cudaDeviceSynchronize();	

		render();
	}

	checkError();
	
	for (int i = 0; i < numParticles; i++) {
		printf("Particle %d: (%f, %f, %f)\n", i, particles[i].position.x, particles[i].position.y, particles[i].position.z);
		//printf("Particle %d: (%f, %f, %f)\n", i, physicsOps[i].velocity.x, physicsOps[i].velocity.y, physicsOps[i].velocity.z);
	}
	
	glutMainLoop();


	return 0;
}


//https://www.nvidia.com/content/gtc/documents/1055_gtc09.pdf OpenGL help
