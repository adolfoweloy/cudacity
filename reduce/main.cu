#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "reduce.h"

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: reduce <input_file>\n");
        exit(-1);
    }

    // retrieving the file name
    char *path;
    size_t list_size = 0;
    path = (char*) malloc(sizeof(char) * strlen(argv[1]));
    strcpy(path, argv[1]);

    // creating matrices from file
    FILE *h_file = fopen(path, "r");
    fscanf(h_file, "%d", &list_size);

    // host defs
    size_t h_list[list_size][N * N];
    size_t h_output[N * N];

    for (size_t i = 0; i < list_size; ++i)
    {
        fscanf(h_file, "%*s");
        for (size_t j = 0; j < N * N; ++j)
            fscanf(h_file, "%lu", &h_list[i][j]);
    }
    fclose(h_file);

    // debug
    printf("host matrices\n");
    for (size_t i = 0; i < list_size; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            for (size_t k=0; k < N; ++k)
                printf("%lu ", h_list[i][j * N + k]);
            printf("\n");
        }
        printf("\n");
    }
    
    reduce(list_size, (size_t **)h_list, h_output);

    // print result
    printf("\n");
    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < N; ++j)
            printf("%lu ", h_output[i * N + j]);
        printf("\n");
    }

    printf("\n");

    return 0;
}

