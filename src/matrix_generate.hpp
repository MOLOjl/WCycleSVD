#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>

using namespace std;

void generate_matrix(int height, int width)
{
	double temp;
	srand(time(NULL));
	
	string path = "./data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";

	FILE* fp = fopen(path.data(), "w");

	for (int i = 0; i < height * width; i++)
	{
		temp = rand() % 100000 / (float)100000;
		fprintf(fp, "%lf ", temp);  
	}

	fclose(fp);
	// printf("Matrix A(%dx%d) has been generated in %s\n", height, width, path.data());
}