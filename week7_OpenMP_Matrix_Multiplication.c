/*objective: matrix multiplication with and without OpenMP */
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

int main(int argc, char *argv[]) {

	int **a, **b, **c; // variables to store allocated memory
	int a_r, a_c, b_r, b_c; // variables to input matrix size
	int nthreads, tid, chunk = 10; //variables to be used by OpenMP functions
	double start_serial, end_serial; //variables to store time for serial execution
	double start_parallel, end_parallel; //variables to store time for parallel execution
	double dif_serial, dif_parallel; //variable to calculate the time difference
	int i, j, k; // variables to be used in for loops to generate matrices

	// get the matrix dimensions from user
	again: printf("\nenter rows and columns for matrix one:");
	scanf("%d%d", &a_r, &a_c);
	printf("\nenter rows and columns for matrix two:");
	scanf("%d%d", &b_r, &b_c);
	if (a_c != b_r) {
		printf("\ncan not multiply");
		goto again;
	}

	// multiply matrices without using OpenMP
	printf("******************************************************\n");
	printf("Without using OpenMP to multiply matrices \n");

	/* allocate memory for matrix one */
	a = (int **) malloc(10 * a_r);
	for (i = 0; i < a_c; i++) {
		a[i] = (int *) malloc(10 * a_c);
	}

	/* allocate memory for matrix two */
	b = (int **) malloc(10 * b_r);
	for (i = 0; i < b_c; i++) {
		b[i] = (int *) malloc(10 * b_c);
	}

	/* allocate memory for sum matrix */
	c = (int **) malloc(10 * a_r);
	for (i = 0; i < b_c; i++) {
		c[i] = (int *) malloc(10 * b_c);
	}

	printf("Initializing matrices...\n");

	start_serial = omp_get_wtime();; //start the timer

	//initializing first matrix
	for (i = 0; i < a_r; i++) {
		for (j = 0; j < a_c; j++) {
			a[i][j] = i + j;
		}
	}

	// initializing second matrix
	for (i = 0; i < b_r; i++) {
		for (j = 0; j < b_c; j++) {
			b[i][j] = i * j;
		}
	}

	// initialize product matrix
	for (i = 0; i < a_r; i++) {
		for (j = 0; j < b_c; j++) {
			c[i][j] = 0;
		}
	}

	// multiply matrix one and two
	for (i = 0; i < a_r; i++) {
		for (j = 0; j < a_c; j++) {
			for (k = 0; k < b_c; k++) {
				c[i][j] = c[i][j] + a[i][k] * b[k][j];
			}
		}
	}

	end_serial = omp_get_wtime();; //end the timer

	dif_serial = end_serial - start_serial; //store the difference

	printf("matrix multiplication without using OpenMP took %f sec. time.\n",
			dif_serial);

	/*free memory*/
	for (i = 0; i < a_r; i++) {
		free(a[i]);
	}
	free(a);

	for (i = 0; i < a_c; i++) {
		free(b[i]);
	}
	free(b);

	for (i = 0; i < b_c; i++) {
		free(c[i]);
	}
	free(c);

	printf("matrix multiplication without using OpenMP ended\n");

	printf("******************************************************\n");

	printf("Using OpenMP to multiply matrices \n");

	/* allocate memory for matrix one */
	a = (int **) malloc(10 * a_r);
	for (i = 0; i < a_c; i++) {
		a[i] = (int *) malloc(10 * a_c);
	}

	/* allocate memory for matrix two */
	b = (int **) malloc(10 * b_r);
	for (i = 0; i < b_c; i++) {
		b[i] = (int *) malloc(10 * b_c);
	}

	/* allocate memory for sum matrix */
	c = (int **) malloc(10 * a_r);
	for (i = 0; i < b_c; i++) {
		c[i] = (int *) malloc(10 * b_c);
	}

	printf("Initializing matrices...\n");

	start_parallel = omp_get_wtime(); //start the timer

	/*** Spawn a parallel region explicitly scoping all variables ***/
	#pragma omp parallel shared(a,b,c,nthreads,chunk) private(tid,i,j,k)
	{
		tid = omp_get_thread_num();
		if (tid == 0) {
			nthreads = omp_get_num_threads();
			printf("Starting matrix multiplication with %d threads\n",
					nthreads);
		}

		//initializing first matrix
		#pragma omp for schedule (static, chunk)
		for (i = 0; i < a_r; i++) {
			for (j = 0; j < a_c; j++) {
				a[i][j] = i + j;
			}
		}

		// initializing second matrix
		#pragma omp for schedule (static, chunk)
		for (i = 0; i < b_r; i++) {
			for (j = 0; j < b_c; j++) {
				b[i][j] = i * j;
			}
		}

		// initialize product matrix
		#pragma omp for schedule (static, chunk)
		for (i = 0; i < a_r; i++) {
			for (j = 0; j < b_c; j++) {
				c[i][j] = 0;
			}
		}

		/*** Do matrix multiply sharing iterations on outer loop ***/
		/*** Display who does which iterations for demonstration purposes ***/
		printf("Thread %d starting matrix multiplication...\n", tid);

		#pragma omp for schedule (static, chunk)
		for (i = 0; i < a_r; i++) {
			// printf("Thread=%d did row=%d\n",tid,i);
			for (j = 0; j < a_c; j++) {
				for (k = 0; k < b_c; k++) {
					c[i][j] = c[i][j] + a[i][k] * b[k][j];
				}
			}
		}
	} /***** end of parallel region *****/

	printf("Done.\n");
	end_parallel = omp_get_wtime();    //end the timer
	dif_parallel = end_parallel - start_parallel; //store the difference
	printf("Matrix multiplication using OpenMP took %f sec. time.\n",
			dif_parallel);
	printf("matrix multiplication without using OpenMP ended\n");
	printf("******************************************************\n");

	/*free memory*/
	for (i = 0; i < a_r; i++) {
		free(a[i]);
	}
	free(a);

	for (i = 0; i < a_c; i++) {
		free(b[i]);
	}
	free(b);

	for (i = 0; i < b_c; i++) {
		free(c[i]);
	}
	free(c);
}
