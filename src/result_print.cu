#ifndef RESULT_PRINT
#define RESULT_PRINT

#include <string>
#include <iostream>

void print_matrix(double* m, int h, int w, std::string path){
    FILE* fp;
    fp = fopen(path.c_str(), "w");
    // fp = fopen(path.c_str(), "a+");
    
    // row major
    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            fprintf(fp, "%lf ", m[i * w + j]);
        }
        fprintf(fp, "\n");
    }
    
    // column major
    // for(int i=0; i<h; i++){
    //     for(int j=0; j<w; j++){
    //         fprintf(fp, "%lf ", m[j * h + i]);
    //     }
    //     fprintf(fp, "\n");
    // }
    
    fprintf(fp, "\n");
    fclose(fp);
}

void print_int_matrix(unsigned int* m, int h, int w, std::string path){
    FILE* fp;
    fp = fopen(path.c_str(), "a+");

    for(int i=0; i<h; i++){
        for(int j=0; j<w; j++){
            fprintf(fp, "%d ", m[i * w + j]);
        }
        fprintf(fp, "\n");
    }

    fprintf(fp, "\n");
    fclose(fp);
}

#endif