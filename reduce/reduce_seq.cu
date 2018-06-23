#include <stdio.h>
#include <stdlib.h>
#include "reduce_seq.h"

void reduce_seq(size_t list_size, size_t h_list[][M], size_t *h_output)
{
    for (int i=0; i<M; ++i)
        h_output[i] = INT_MAX;

    for (int col=0; col<M; ++col)
    {
        for (int i=0; i<list_size; ++i)
        {
            if (h_list[i][col] < h_output[col])
                h_output[col] = h_list[i][col];
        }
    }
}

