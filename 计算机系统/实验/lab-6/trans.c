/* 
 * trans.c - Matrix transpose B = A^T
 *
 * Each transpose function must have a prototype of the form:
 * void trans(int M, int N, int A[N][M], int B[M][N]);
 *
 * A transpose function is evaluated by counting the number of misses
 * on a 1KB direct mapped cache with a block size of 32 bytes.
 */ 
#include <stdio.h>
#include "cachelab.h"

int is_transpose(int M, int N, int A[N][M], int B[M][N]);

/* 
 * transpose_submit - This is the solution transpose function that you
 *     will be graded on for Part B of the assignment. Do not change
 *     the description string "Transpose submission", as the driver
 *     searches for that string to identify the transpose function to
 *     be graded. 
 */
char transpose_submit_desc[] = "Transpose submission";
void transpose_submit(int M, int N, int A[N][M], int B[M][N])
{
 if(M==32&&N==32)
    {
        for(int i=0;i<M;i=i+8)
        {
            for(int j=0;j<N;j++)
            {
                int temp1=A[j][i];
                int temp2=A[j][i+1];
                int temp3=A[j][i+2];
                int temp4=A[j][i+3];
                int temp5=A[j][i+4];
                int temp6=A[j][i+5];
                int temp7=A[j][i+6];
                int temp8=A[j][i+7];
                B[i][j]=temp1;
                B[i+1][j]=temp2;
                B[i+2][j]=temp3;
                B[i+3][j]=temp4;
                B[i+4][j]=temp5;
                B[i+5][j]=temp6;
                B[i+6][j]=temp7;
                B[i+7][j]=temp8;
            }
        }
    }
     if(M==64&&N==64)
    {
         for(int i=0;i<M;i+=8)
        {
            for(int j=0;j<N;j+=8)
            {
                for(int k=i;k<i+4;k++)
                {
                    int temp1=A[k][j];
                    int temp2=A[k][j+1];
                    int temp3=A[k][j+2];
                    int temp4=A[k][j+3];
                    int temp5=A[k][j+4];
                    int temp6=A[k][j+5];
                    int temp7=A[k][j+6];
                    int temp8=A[k][j+7];
                    B[j][k]=temp1;
                    B[j+1][k]=temp2;
                    B[j+2][k]=temp3;
                    B[j+3][k]=temp4;
                    B[j][k+4]=temp5;
                    B[j+1][k+4]=temp6;
                    B[j+2][k+4]=temp7;
                    B[j+3][k+4]=temp8;
                }
                for(int m=j;m<j+4;m++)
                {
                    int temp1=A[i+4][m];
                    int temp2=A[i+5][m];
                    int temp3=A[i+6][m];
                    int temp4=A[i+7][m];
                    int temp5=B[m][i+4];
                    int temp6=B[m][i+5];
                    int temp7=B[m][i+6];
                    int temp8=B[m][i+7];
                    B[m][i+4]=temp1;
                    B[m][i+5]=temp2;
                    B[m][i+6]=temp3;
                    B[m][i+7]=temp4;
                    B[m+4][i]=temp5;
                    B[m+4][i+1]=temp6;
                    B[m+4][i+2]=temp7;
                    B[m+4][i+3]=temp8;
                }
                for(int k=i+4;k<i+8;k++)
                {
                    int temp1=A[k][j+4];
                    int temp2=A[k][j+5];
                    int temp3=A[k][j+6];
                    int temp4=A[k][j+7];
                    B[j+4][k]=temp1;
                    B[j+5][k]=temp2;
                    B[j+6][k]=temp3;
                    B[j+7][k]=temp4;
                }                
            }
        }
    }
     if(M==61)
    {
        for(int i=0;i<M;i+=17)
        {
            for(int j=0;j<N;j+=17)
            {
                for(int k=i;k<i+17&&k<N;k++)
                {
                    for(int m=j;m<j+17&&m<M;m++)
                    {
                        B[m][k]=A[k][m];
                    }
                }
            }
        }
    }
}

/* 
 * You can define additional transpose functions below. We've defined
 * a simple one below to help you get started. 
 */ 

/* 
 * trans - A simple baseline transpose function, not optimized for the cache.
 */
char trans_desc[] = "Simple row-wise scan transpose";
void trans(int M, int N, int A[N][M], int B[M][N])
{
    int i, j, tmp;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            tmp = A[i][j];
            B[j][i] = tmp;
        }
    }    

}

/*
 * registerFunctions - This function registers your transpose
 *     functions with the driver.  At runtime, the driver will
 *     evaluate each of the registered functions and summarize their
 *     performance. This is a handy way to experiment with different
 *     transpose strategies.
 */
void registerFunctions()
{
    /* Register your solution function */
    registerTransFunction(transpose_submit, transpose_submit_desc); 

    /* Register any additional transpose functions */
    registerTransFunction(trans, trans_desc); 

}

/* 
 * is_transpose - This helper function checks if B is the transpose of
 *     A. You can check the correctness of your transpose by calling
 *     it before returning from the transpose function.
 */
int is_transpose(int M, int N, int A[N][M], int B[M][N])
{
    int i, j;

    for (i = 0; i < N; i++) {
        for (j = 0; j < M; ++j) {
            if (A[i][j] != B[j][i]) {
                return 0;
            }
        }
    }
    return 1;
}

