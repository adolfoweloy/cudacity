#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "reduce.h"
#include "reduce_seq.h"

void assert_m(size_t *h_output, size_t *h_output_seq)
{
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
        {
            if (h_output[i * N + j] != h_output_seq[i * N + j])
            {
                printf("test fail: Matrices aren't equal\n");
                return;
            }
        } 
    printf("test passed!\n");
}

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
    size_t h_output_seq[N * N];

    for (size_t i = 0; i < list_size; ++i)
    {
        fscanf(h_file, "%*s");
        for (size_t j = 0; j < N * N; ++j)
            fscanf(h_file, "%lu", &h_list[i][j]);
    }
    fclose(h_file);

    reduce(list_size, (size_t **)h_list, h_output);
    reduce_seq(list_size, h_list, h_output_seq);

    assert_m(h_output, h_output_seq);
 
    printf("\n");

    return 0;
}

