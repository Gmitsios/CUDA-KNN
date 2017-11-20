/*Parallel and Distributed Systems
---3rd Assignment---
--CUDA KNN Algorithm--
-Author: Mitsios Georgios
-AEM: 6976
-September 2014
*/


#include <stdio.h>
#include <stdlib.h>
#include  <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>


#define CUDA_CHECK_RETURN(value) {								\
	cudaError_t _m_cudaStat = value;							\
	if (_m_cudaStat != cudaSuccess) {							\
		fprintf(stderr, "Error %s at line %d in file %s\n",				\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);									\
	} }

typedef struct{
	double *data;
	int leading_dim;
	int secondary_dim;
} knn_struct;

//1st input option. Random initialization
void random_initialization(knn_struct *set, int cal){

  int i = 0;
  int n = set->leading_dim;
  int m = set->secondary_dim;
  double *tmp_set = set->data;

  srand(cal*time(NULL));
  /*Generate random floating points [-50 50]*/
  for(i=0; i<m*n; i++){
    tmp_set[i] = 100 * (double)rand() / RAND_MAX - 50;
  }

}

//2nd input option. Import data from benchmark files
void initialize(knn_struct *set){

  int i = 0;
  int n = set->leading_dim;
  int m = set->secondary_dim;
  float *tmp_set;
  double *tempArray;
 
  tmp_set = (float*)malloc(n*m*sizeof(float));
  tempArray = (double*)malloc(n*m*sizeof(double));

  FILE *fp;
  size_t t;

  if (m>100000){
  	if ((fp = fopen("baseREAD.bin", "rb")) == NULL){ printf("Can't open output file"); }
	}
  else{ if ((fp = fopen("queriesREAD.bin", "rb")) == NULL){ printf("Can't open output file"); } }

  t = fread(tmp_set, n*m, sizeof(float),  fp);
  
  fclose(fp);
  //Convert float input data to doubles
  for (i=0;i<n*m;i++){
	tempArray[i] = (double)tmp_set[i];
  }
  set->data = tempArray;
}

//This function was used to normalize data from -50 to 50 to match the random points
  void input_normalisation(knn_struct *base, knn_struct *queries){
	int i;
	int N = base->leading_dim, ni = queries->leading_dim;
  	int M = base->secondary_dim, mi = queries->secondary_dim;
	double maxVal=0, minVal=1;

	double *tmp_data = base->data;
  	double *tmp_queries = queries->data;

	for (i=0;i<N*M;i++){
		if (tmp_data[i]>maxVal) { maxVal=tmp_data[i]; }
		if (tmp_data[i]<minVal) { minVal=tmp_data[i]; }
  	}

	for (i=0;i<ni*mi;i++){
		if (tmp_queries[i]>maxVal) { maxVal=tmp_queries[i]; }
		if (tmp_queries[i]<minVal) { minVal=tmp_queries[i]; }
  	}

  	if ( minVal<0 ) {  minVal = minVal*(-1);   }
  	if ( minVal>maxVal) { maxVal = minVal; }

  	for (i=0;i<N*M;i++){
		tmp_data[i] = (tmp_data[i]*50)/maxVal;
  	}

	for (i=0;i<ni*mi;i++){
		tmp_queries[i] = (tmp_queries[i]*50)/maxVal;
  	}
  }

void save_d(double* data, char* file, int N, int M){

  FILE *outfile;

  printf("Saving data to file: %s\n", file);

  if((outfile=fopen(file, "wb")) == NULL){
    printf("Can't open output file");
  }

  fwrite(data, sizeof(double), N*M, outfile);

  fclose(outfile);

}

void save_int(int* data, char* file, int N, int M){

  FILE *outfile;

  printf("Saving data to file: %s\n", file);

  if((outfile=fopen(file, "wb")) == NULL){
    printf("Can't open output file");
  }

  fwrite(data, sizeof(int), N*M, outfile);

  fclose(outfile);

}

void clean(knn_struct* d){
  free(d->data);
}

//This kernel function computes the euclidean distance between queries and objects
__global__ void calculate_distance(double* queries, double* dataset, double *dist, int* k, int* n){
	int numObj=*n;
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	double tmp=0;

	//Initialize distances to 0
	if (threadIdx.x == 0) {
		for (int i=0; i<numObj; i++){
			dist[blockIdx.x*numObj + i] = 0;
		}
	}
	__syncthreads();

	//Compute and store euclidean distance
	for (int ni=0;ni<(numObj);ni++){
		tmp = (queries[index] - dataset[threadIdx.x + ni * blockDim.x])*(queries[index] - dataset[threadIdx.x + ni * blockDim.x]);
		dist[blockIdx.x * numObj + ni] = dist[blockIdx.x * numObj + ni] + tmp;
		__syncthreads();
	}
	__syncthreads();
}

/*This kernel function computes the knn of the temporary set of data (which is only a portion of the 
original data. The way this function works is better expalined in the report sheet*/
__global__ void compute_knn(double* NNdist, double* dist, int* numObj, int* NNidx, int* bonusID, int *offset, double *tmpDist, int *tmpID, int *N){

	int i, n = *numObj, off = *offset, Nol= *N;
	int start = blockIdx.x * n;
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	int position = blockDim.x+threadIdx.x+1;
	double temp=0;
	int tmp=0;

	NNdist[index] = DBL_MAX;
	NNidx[index] = (-1);
	bonusID[index] = (-1);

	__syncthreads();

	for (i=start; i<(start+n+blockDim.x);i++){
		if ( ((i-position)< (start +n)) && (i-position) >= start ){

			if ( NNdist[index] > dist[i - position] ){
				temp = NNdist[index];
				NNdist[index] = dist[i - position];
				dist[i - position] = temp;

				if (bonusID[index]>=0){
					tmp = NNidx[index];
					NNidx[index]=bonusID[index];
					bonusID[index]= tmp;
				}
				else{
					bonusID[index] = NNidx[index];
					NNidx[index] = (i - position - start) + (off*n);
				}

			}
			__syncthreads();
		}
		__syncthreads();

		if (threadIdx.x == 7){
			for (int j=0;j<(blockDim.x-1);j++){
				bonusID[j+blockIdx.x*blockDim.x] = bonusID[j+blockIdx.x*blockDim.x +1];
			}
			bonusID[index] = (-1);
		}
		__syncthreads();

	}
	tmpDist[ off*blockDim.x + blockIdx.x*(Nol*blockDim.x) + threadIdx.x ] = NNdist[index];
	tmpID[ off*blockDim.x + blockIdx.x*(Nol*blockDim.x) + threadIdx.x ] = NNidx[index];
	__syncthreads();
}

/*This is the last kernel function which compares the knns that have been computed from each group
of data to find the final knn values and ids. The operation is similar to the "compute_knn" function*/
__global__ void knnSelection(double* NNdist, int* NNidx, int* originalID, double* dist, int* numObj){
	int i, n=*numObj;
	int start = blockIdx.x * n;
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	double temp=0;
	int tmp=0;

	NNdist[index] = DBL_MAX;
	NNidx[index] = (-1);

	__syncthreads();

	for (i=start; i<(start+n+blockDim.x);i++){
		if ( ((i-blockDim.x+threadIdx.x+1)< (start +n)) && (i-blockDim.x+threadIdx.x+1) >= start ){
			if ( NNdist[index] > dist[i - blockDim.x + threadIdx.x + 1] ){

				temp = NNdist[index];
				NNdist[index] = dist[i - blockDim.x + threadIdx.x + 1];
				dist[i - blockDim.x + threadIdx.x + 1] = temp;

				tmp = NNidx[index];
				NNidx[index]=originalID[i - blockDim.x + threadIdx.x + 1];
				originalID[i - blockDim.x + threadIdx.x + 1]= tmp;

			}
		}
		__syncthreads();
	}
	__syncthreads();

}


int main(int argc, char **argv){

	struct timeval first, second, lapsed;
	struct timezone tzp;
	int i;

	int numObjects = atoi(argv[1]); //pow(2,atoi(argv[1]));
	int numDim = atoi(argv[2]);
	int numQueries = atoi(argv[3]);
	int k = atoi(argv[4]);

	int pi=pow(2,15);
	int pol;
	if (numObjects>pi ){pol=numObjects/pi;}
	else{ pol=1; pi=numObjects;	}

	//Getting GPU memory's info
	size_t mem_tot_0 = 0;
	size_t mem_free_0 = 0;
	cudaMemGetInfo(&mem_free_0, &mem_tot_0);
	size_t dist_size = numQueries*numObjects*sizeof(double);
	size_t data_size = numObjects*numDim*sizeof(double);
	size_t queries_size = numQueries*numDim*sizeof(double);
	size_t NNdist_size = numQueries*k*sizeof(double);
	size_t NNidx_size = numQueries*k*sizeof(int);
	size_t mem_req = dist_size + data_size + queries_size + NNdist_size + NNidx_size;
	size_t mem_req1 = dist_size + data_size;
	int nol;
	int ni;
	size_t mem_free_n = mem_free_0 - (queries_size -NNdist_size - NNidx_size);
	printf("\nGPU memory needed : %ld bytes \n", mem_req1);
	printf("Available GPU memory  : %ld bytes \n", mem_free_n);

	//Computing the number of divisions to the data for proper memory usage
	if (mem_req> (mem_free_n ) ){
		nol = (mem_req1/mem_free_n);
		int e=0;
		while (nol>2){
			nol=nol/2;
			e++;
		}
		nol=pow(2,e+1);
	}
	else {nol=1;}
	//Value 500 can be reduced to 200 or even 100 for lower memory GPUs
	if (numQueries>500) { nol = nol*(numQueries/500); }
	ni = numObjects/nol;
	printf("Data will be divided into %d groups \n", nol);
	printf("of %d elements per group for optimal operation\n\n", ni);

	char *dataset_file = "training_set.bin";
	char *query_file = "query_set.bin";
	char *KNNdist_file = "KNNdist.bin";
	char *KNNidx_file = "KNNidx.bin" ;
	double *tmpDist;
	int *tmpID;
	tmpDist = (double*)malloc(nol*numQueries*k*sizeof(double));
	tmpID = (int*)malloc(nol*numQueries*k*sizeof(int));

	printf("objects: %d\n", numObjects);
	printf("dimentions: %d\n", numDim);
	printf("queries: %d\n", numQueries);
	printf("k: %d\n", k);

	knn_struct training_set;
	knn_struct query_set;
	double *dist;
	double *NNdist;
	int *NNidx;

	training_set.leading_dim = numDim;
	training_set.secondary_dim = numObjects;
	query_set.leading_dim = numDim;
	query_set.secondary_dim = numQueries;

	/*======== Memory allocation ======*/
	training_set.data = (double*)malloc(numObjects*numDim*sizeof(double));
	query_set.data = (double*)malloc(numQueries*numDim*sizeof(double));
	NNdist = (double*)malloc(numQueries*k*sizeof(double));
	NNidx = (int*)malloc(numQueries*k*sizeof(int));
	dist = (double*)malloc(numObjects*numQueries*sizeof(double));

	double *d_data;
	double *d_queries;
	double *d_dist;
	double *d_NNdist;
	int *d_NNidx;
	int *d_bonusID;
	int *dev_k, *dev_n;
	int *dev_nol;
	double *d_tmpDist;
	int *d_tmpID;

	//GPU memory allocation
	CUDA_CHECK_RETURN(cudaMalloc( (void **)&d_data, ni*numDim*sizeof(double)) );
	CUDA_CHECK_RETURN(cudaMalloc( (void **)&d_queries, numQueries*numDim*sizeof(double)) );

	CUDA_CHECK_RETURN( cudaMalloc( (void **)&d_NNdist, numQueries*k*sizeof(double)) );
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&d_NNidx, numQueries*k*sizeof(int)) );

	CUDA_CHECK_RETURN( cudaMalloc( (void **)&d_bonusID, numQueries*k*sizeof(int)) );
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&d_tmpDist, nol*numQueries*k*sizeof(double)) );
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&d_tmpID, nol*numQueries*k*sizeof(int)) );

	CUDA_CHECK_RETURN( cudaMalloc( (void **)&d_dist, numQueries*ni*sizeof(double)) );

	CUDA_CHECK_RETURN( cudaMalloc( (void **)&dev_k, sizeof(int)) );
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&dev_n, sizeof(int)) );
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&dev_nol, sizeof(int)) );

	/*======== initialize =========*/

	//Input option 1. Create random data
	random_initialization(&training_set, 1);
	random_initialization(&query_set, 2);

	//Input option 2. Import data from file
	//initialize(&training_set);
	//initialize(&query_set);

	/*The following function was used to normalize imported data values from benchmark files 
	to be from -50 to 50, because there was some undefined error in previous compilations. 
	It MAY OR MAY NOT BE USED */
	//input_normalisation(&training_set, &query_set);

	//Copy data to GPU memory
	CUDA_CHECK_RETURN(cudaMemcpy( d_queries, query_set.data, numQueries*numDim*sizeof(double), cudaMemcpyHostToDevice ) );
	CUDA_CHECK_RETURN(cudaMemcpy( dev_k, &k, sizeof(int), cudaMemcpyHostToDevice ) );
	CUDA_CHECK_RETURN(cudaMemcpy( dev_nol, &nol, sizeof(int), cudaMemcpyHostToDevice ) );
	CUDA_CHECK_RETURN(cudaMemcpy( dev_n, &ni, sizeof(int), cudaMemcpyHostToDevice ) );
	CUDA_CHECK_RETURN(cudaMemcpy( d_tmpDist, tmpDist, nol*numQueries*k*sizeof(double), cudaMemcpyHostToDevice ) );
	CUDA_CHECK_RETURN(cudaMemcpy( d_tmpID, tmpID, nol*numQueries*k*sizeof(int), cudaMemcpyHostToDevice ) );
	CUDA_CHECK_RETURN(cudaMemcpy( d_dist, dist, numQueries*ni*sizeof(double), cudaMemcpyHostToDevice ) );


	int offset=0;
	int *dev_off=0;
	CUDA_CHECK_RETURN( cudaMalloc( (void **)&dev_off, sizeof(int)) );

	gettimeofday(&first, &tzp);

	/*This for loop is the core of the program. It sends each group of data to the GPU 
	where it computes the distances between each query and each object. Then calls the
	"compute_knn" kernel function to find the actual KNNs*/
	for ( offset=0 ; offset<nol ; offset++ ){
		CUDA_CHECK_RETURN(cudaMemcpy( d_data, (training_set.data +offset*ni), ni*numDim*sizeof(double), cudaMemcpyHostToDevice ) );
		calculate_distance<<< numQueries, numDim>>> ( d_queries, d_data, d_dist, dev_k, dev_n);
		CUDA_CHECK_RETURN( cudaPeekAtLastError() );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );

		CUDA_CHECK_RETURN(cudaMemcpy( dev_off, &offset, sizeof(int), cudaMemcpyHostToDevice ) );
		compute_knn<<< numQueries, k>>>(d_NNdist, d_dist, dev_n, d_NNidx, d_bonusID, dev_off, d_tmpDist, d_tmpID, dev_nol);
		CUDA_CHECK_RETURN( cudaPeekAtLastError() );
		CUDA_CHECK_RETURN( cudaDeviceSynchronize() );
	}

	CUDA_CHECK_RETURN(cudaMemcpy( tmpDist, d_tmpDist, nol*numQueries*k*sizeof(double), cudaMemcpyDeviceToHost ) );
	CUDA_CHECK_RETURN(cudaMemcpy( tmpID, d_tmpID, nol*numQueries*k*sizeof(int), cudaMemcpyDeviceToHost ) );

	/*In case of division of the original data, the temporary KNNs are being compared
	to form the final KNNs*/
	if (nol!=1){
		pi=nol*k;
		cudaMemcpy( dev_n, &pi, sizeof(int), cudaMemcpyHostToDevice ) ;

		knnSelection<<< numQueries, k>>>(d_NNdist, d_NNidx, d_tmpID, d_tmpDist, dev_n);
		CUDA_CHECK_RETURN( cudaPeekAtLastError() );
	}

	gettimeofday(&second, &tzp);
	printf("\n---KNN computed--- \n\n");
	if(first.tv_usec>second.tv_usec){
		second.tv_usec += 1000000;
		second.tv_sec--;
	}

	lapsed.tv_usec = second.tv_usec - first.tv_usec;
	lapsed.tv_sec = second.tv_sec - first.tv_sec;

	printf("Time elapsed: %ld, %ld s\n", lapsed.tv_sec, lapsed.tv_usec);


	CUDA_CHECK_RETURN( cudaMemcpy( NNdist, d_NNdist, numQueries*k*sizeof(double), cudaMemcpyDeviceToHost ) );
	CUDA_CHECK_RETURN( cudaMemcpy( NNidx, d_NNidx, numQueries*k*sizeof(int), cudaMemcpyDeviceToHost ) );
	
	save_d(query_set.data, query_file, numQueries, numDim);
	save_d(training_set.data, dataset_file, numObjects, numDim);
	save_d(NNdist, KNNdist_file, k, numQueries);
	save_int(NNidx, KNNidx_file, k, numQueries);

	
	/*===== clean memory ========*/
	clean(&training_set);
	clean(&query_set);
	free(NNdist);
	free(NNidx);
	free(dist);

	cudaFree(d_queries);
	cudaFree(d_data);
	cudaFree(d_NNdist);
	cudaFree(d_NNidx);
	cudaFree(dev_k);
	cudaFree(dev_n);

	cudaFree(d_dist);

}
