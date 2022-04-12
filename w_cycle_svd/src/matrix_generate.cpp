#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>

using namespace std;

void generate_matrix(int height, int width)
{
	double temp;
	srand(time(NULL));

	string path = "../data/generated_matrixes/A_h" + to_string(height) + "_w" + to_string(width)+ ".txt";

	FILE* fp = fopen(path.data(), "w");

	for (int i = 0; i < height * width; i++)
	{
		temp = rand() % 100000 / (float)100000;
		fprintf(fp, "%lf ", temp);  
	}
	printf("size:%d, v:%lf\n", height*width, temp);
	printf("Matrix A(%dx%d) has been generated in %s\n", height, width, path.data());
}

int main(int argc, char* argv[]){
	int height, width;
	if(argc == 3){
		sscanf(argv[1], "%d", &height);
		sscanf(argv[2], "%d", &width);		
	}
	else{
		printf("???\n");
		return 0;
	}
	generate_matrix(height, width);
	return 0;
}