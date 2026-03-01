// Rel_07  2024
#include "TYPES.h"

#include <stdio.h>

#include <stdlib.h>

#include <assert.h>

#include <cuda_runtime.h>
 //#include <helper_cuda.h>

#include <cstdlib>

#include <cstring>

#include <cmath>

#include <ctime>

#include <set>

#include <iostream>

#include <fstream>

#include <string>

using namespace std;
cudaError_t error;
ifstream infile;

#include "GraphRP.h"

#define MaxNBL 1024 // GeForce GT 620M has 2 x 48 cores & GTX 980 has 16 x (4 x 32) cores
#define NTH 32 // Tesla 40C has 15 x 192 = 2880 
#define MaxN 90 // maximum Number of nodes; O.W. Use dynamic Allocation to shared pointers
const int NGpus = 1; // Number of GPUs
int NBL;
typedef unsigned char * MASKTYPE;
#define MAX_Q_SIZE 1000
Graph_Type * d_Graph[NGpus];
N_Type * d_Nodes[NGpus];
E_Type * d_Edges[NGpus];
double * Prob, * d_Prob[NGpus];
int * d_BNP[NGpus];
int BNP[NGpus][MaxNBL];
double * d_Brel[NGpus], * d_BEps[NGpus]; // DEVICE BLOCK RELIABILITY & Epsilon
double Brel[NGpus][MaxNBL], BEps[NGpus][MaxNBL]; // Reliability & Epsilon computed by each block ??? GPU ???
int N; //number of nodes
int M; //number of edges
unsigned char GT;
int src, t; //  source node & destination 

set < int > target_set; // used for K-terminal Reliability

/*__________________________________________________________________________________*/
__device__ void CopyMask(MASKTYPE MASKQUE[], double MKEY[], int from, int to, int MS) {
  for (int i = 0; i < MS; i++) MASKQUE[to][i] = MASKQUE[from][i];
  MKEY[to] = MKEY[from];
  return;
}
__device__ double PURGEQ(int MQXFront, int MQXRear, double MKEY[]) {
  double SumQ = 0;
  for (int i = MQXFront; i <= MQXRear; i++) SumQ += MKEY[i];
  return SumQ;
}
__device__ void QDUMP(MASKTYPE MASKQUE[], double MKEY[], int MQXFront, int MQXRear, int MS) {
  int i = MQXFront, j = MQXRear, QS = j - i + 1;;
  printf(" QS = %d\n", QS);
  for (int x = i; x <= j; x++) {
    printf(" MASKQUE[%d] %p : ", x, MASKQUE[x]);
    for (int k = 0; k < MS; k++) printf(" %x ", MASKQUE[x][k]);
    printf(" MKEY = %lf \n", MKEY[x]);
  }
  return;
}

__device__ void __inline__ REHEAPIFY(MASKTYPE * MASKQUE, double MKEY[], MASKTYPE TempM, int n) { // From root down
  double TempK;
  if (n <= 1) return;
  int i = 0, i1, i2, CX;
  while (i < n / 2) {
    i1 = 2 * i + 1;
    i2 = 2 * i + 2;
    CX = i1;
    if (i2 < n)
      if (MKEY[i2] > MKEY[i1]) CX = i2;
    if (MKEY[CX] > MKEY[i]) {
      TempM = MASKQUE[CX];
      TempK = MKEY[CX];
      MASKQUE[CX] = MASKQUE[i];
      MKEY[CX] = MKEY[i];
      MASKQUE[i] = TempM;
      MKEY[i] = TempK;
      i = CX;
    } else return;
  }
}

__device__ void __inline__ HEAPIFY(MASKTYPE * MASKQUE, double MKEY[], MASKTYPE TempM, int n) { // From down up
  double TempK;
  if (n <= 1) return;
  int i = n - 1, PX;
  while (i > 0) {
    PX = (i - 1) / 2;
    if (MKEY[PX] < MKEY[i]) {
      TempM = MASKQUE[PX];
      TempK = MKEY[PX];
      MASKQUE[PX] = MASKQUE[i];
      MKEY[PX] = MKEY[i];
      MASKQUE[i] = TempM;
      MKEY[i] = TempK;
      i = PX;
    } else return;
  }
}

//_____________________________________________________________________________________________________

__device__ void TID_TO_MASK(unsigned char * mask, int tid, int MaskSize, int TID_SIZE) {
  for (int e = 0; e < MaskSize; e++) mask[e] = (unsigned char) 0xFF; // UNMARKED EDGE
  if (TID_SIZE <= MaskSize)
    for (int e = 0; e < TID_SIZE; e++) {
      int m = 1 << e;
      if (tid & m) mask[e] = (unsigned char) 0x1;
      else mask[e] = 0x0;
    }
  else
    for (int e = 0; e < MaskSize; e++) {
      int m = 1 << e;
      if (tid & m) mask[e] = (unsigned char) 0x1;
      else mask[e] = 0x0;
    }
}

__host__ __device__ int Seq_No(int i, int j, N_Type Nodes[], E_Type Edges[]) {
  int offset = Nodes[i].Offset;
  int degree = Nodes[i].degree;
  for (int d = 0; d < degree; d++) {
    E_Type e = Edges[offset + d];
    if (e.dst == j) return (e.SeqNo);
  }
  return 0;
}
__host__ __device__ void PrintSeqNos(int N, int M, N_Type Nodes[], E_Type Edges[]) {
  printf("\n Degrees: ");
  for (int i = 0; i < N; i++) printf(" %d ", Nodes[i].degree);
  printf("\n Offsets: ");
  for (int i = 0; i < N; i++) printf(" %d ", Nodes[i].Offset);
  printf("\n Edges:  ");
  for (int i = 0; i < M; i++) printf(" %d %d %f :", Edges[i].SeqNo, Edges[i].dst, Edges[i].prob);
  printf("\n*********************\n");
}
__device__ __forceinline__ bool Opermask(int N, unsigned char * NewMask, N_Type Nodes[], E_Type Edges[], int src, int t, bool * Visited,
  short Parent[], short Queue[]) {
  for (int i = 0; i < N; i++) Visited[i] = false;
  bool OP = false;
  for (int i = 0; i < N; i++) Parent[i] = -1;

  int front = 0, rear = -1, d, V;
  rear++;
  Queue[rear] = (short) src;
  while (rear >= front) {
    V = Queue[front];
    front++; //printf("V = %d|", V);
    d = Nodes[V].degree;
    int e_l = Nodes[V].Offset;
    for (int dx = 0; dx < d; dx++) {
      int j = Edges[e_l + dx].dst;
      int sn = Edges[e_l + dx].SeqNo;
      //printf("0%xHERE %d %d %d|", V, j, sn);
      if (NewMask[sn] == 1 || NewMask[sn] == 0xFF)
        if (!Visited[j]) {
          Visited[j] = true;
          rear++;
          Queue[rear] = j;
          Parent[j] = V;
        }

      if (Visited[t]) {
        OP = true;
        break;
      }
    } // for
    if (Visited[t]) {
      OP = true;
      break;
    }
  } // while
  return OP;
} // END Opermask

__global__ void TC(Graph_Type * G, N_Type Nodes[], E_Type Edges[], double Prob[], int src, int t, double * d_Brel, double * d_Epsilon, int * d_NP, int TID_SIZE, int Device_ID) {
  __shared__ double rel[NTH];
  __shared__ double Epsilon[NTH];
  __shared__ int NP[NTH]; // number of processed pathsets?
  __shared__ bool Visited[NTH][MaxN]; // BFS src -> t SHOULD I HAVE [NTH] ???
  __shared__ short Parent[NTH][MaxN];
  __shared__ short Queue[NTH][MaxN];//federated learnin fir fault detection
  clock_t start = clock();
  int mTime;
  int M = G -> M, N = G -> N;
  unsigned char GT = G -> GT;
  int sn;
  double MKEY[MAX_Q_SIZE];
  typedef unsigned char * MASKTYPE;
  MASKTYPE TempM;
  int MaskSize;
  double MULT, NewMULT;
  int MQXFront = 0, MQXRear = -1;
  bool OP;

  int e_l;
  int d, dx;
  short V, j;;

  if (GT == 'U') MaskSize = M / 2;
  else MaskSize = M;

  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int TID = Device_ID * gridDim.x * blockDim.x + bx * blockDim.x + tx;
  
  rel[tx] = 0.0;
  Epsilon[tx] = 0.0;
  NP[tx] = 0;
  __syncthreads();
  
  
  if (TID == 0) printf(" %c NNNN=  %d \n", GT, N);

  if (TID == 0) printf(" Grid %d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
  if (TID == 0) {
    printf("DEVICE %d  sf = %d sd = %d \n", Device_ID, (int) sizeof(Prob[0]), (int) sizeof(rel[0]));
  }

  MASKTYPE * MASKQUE = new MASKTYPE[MAX_Q_SIZE]; // MAXIMUM NUMBER OF MASKS ?????

  for (int i = 0; i < MAX_Q_SIZE; i++) {
    MASKQUE[i] = new unsigned char[MaskSize];
  }

  MASKTYPE Mask = new unsigned char[MaskSize]; //// WORKING MASKS ////
  MASKTYPE NewMask = new unsigned char[MaskSize];

  //if (TID == 0) {for (int i = 0; i < MaskSize; i++) printf("%lf * ",Prob[i]); printf("\n");}
  __syncthreads();
  if (TID == 0)
    if (MASKQUE) printf("OKKKKKKKK %p \n", MASKQUE);
  if (MaskSize <= 62)
    if (TID >= ((long int) 2 << (MaskSize - 1))) { //printf("OOPS"); //??????????????????????
      return;
    }
  if (TID == 0) printf("TID SIZE = %d MaskSize = %d MAX_Q_SIZE = %d\n", TID_SIZE, MaskSize, MAX_Q_SIZE);
  //if (TID == 0) PrintSeqNos( N, M, Nodes, Edges);

  TID_TO_MASK(Mask, TID, MaskSize, TID_SIZE); //printf("TID = %d : ", TID); 
  //for (int i = 0; i < MaskSize; i++) printf("%2x*",Mask[i]); printf("\n");
  __syncthreads();

  short * TQ = new short[MaskSize];
  int TQTOP; // STACK TO store sn for unmarked edges ON AN s-t-path -> Rel_08.cu

  //if(TID == 13) {for (int i = 0; i < MaskSize; i++) printf("0x%.2x  ",MASKQUE[0][i]); printf("\n");}
  /*________________START PROCESSING MASK QUEUE ________________________________*/
  MULT = 1.0;
  for (int e = 0; e < MaskSize; e++)
    if (Mask[e] == 1) MULT *= (double) Prob[e];
    else if (Mask[e] == 0) MULT *= (1 - (double) Prob[e]);

  MQXRear++;
  for (int i = 0; i < MaskSize; i++) MASKQUE[MQXRear][i] = Mask[i];
  MKEY[MQXRear] = MULT;
  int Q_SIZE = 1;
  //int T = 0;
  while (Q_SIZE > 0) {
    for (int i = 0; i < MaskSize; i++) Mask[i] = MASKQUE[MQXFront][i];
    MULT = MKEY[MQXFront];
    //printf ("MASK READ FROM Q :");for (int i = 0; i < MaskSize; i++) printf("0x%.2x  ",Mask[i]); printf("\n");
    if (Q_SIZE > 1) {
      CopyMask(MASKQUE, MKEY, MQXRear, MQXFront, MaskSize);
      MQXRear--;
    }
    Q_SIZE -= 1;
    if (Q_SIZE == 0) MQXRear = -1;

    //if (TID == 0) {printf("T = %d BEFORE REHEAPIFY:: ", T++);QDUMP(MASKQUE, MKEY, MQXFront, MQXRear,MaskSize);}
    REHEAPIFY(MASKQUE, MKEY, TempM, Q_SIZE);
    //if (TID == 0) {printf("AFTER:: ");QDUMP(MASKQUE, MKEY, MQXFront, MQXRear,MaskSize);}

    //if(TID == 13) {for (int i = 0; i < MaskSize; i++) printf("0x%.2x  ",Mask[i]); printf("\n");}
    for (int i = 0; i < N; i++) Visited[tx][i] = false;
    OP = false;
    int front = 0, rear = -1;
    for (int i = 0; i < N; i++) Parent[tx][i] = -1;

    rear++;
    Queue[tx][rear] = (short) src;
    while (rear >= front) {
      V = Queue[tx][front];
      front++; //printf("V = %d|", V);
      d = Nodes[V].degree;
      e_l = Nodes[V].Offset;
      for (dx = 0; dx < d; dx++) {
        j = Edges[e_l + dx].dst;
        sn = Edges[e_l + dx].SeqNo; //printf("HERE %d %d %d|", V, j, sn);
        if (Mask[sn] == 1 || Mask[sn] == 0xFF)
          if (!Visited[tx][j]) {
            Visited[tx][j] = true;
            rear++;
            Queue[tx][rear] = j;
            Parent[tx][j] = V;
          }
        if (Visited[tx][t]) {
          OP = true;
          break;
        }
      }
      if (Visited[tx][t]) {
        OP = true;
        break;
      } //if (TID == 13) for (int i = 0; i<N; i++) printf("%d /",Queue[i]); printf("front %d rear %d ///", front, rear);
    } // While

    if (OP) { // Climb up the path from t to s; 
      int i = Parent[tx][t];
      int j = t;
      TQTOP = -1;
      while (j != src) {
        sn = Seq_No(i, j, Nodes, Edges);
        if (Mask[sn] == 0xFF) {
          TQTOP++;
          TQ[TQTOP] = sn;
        }
        j = i;
        i = Parent[tx][j];
      }

      while (TQTOP >= 0) {
        sn = TQ[TQTOP];
        TQTOP--; //Create EXTRA MASKS from TQ
        // cREATE NEW MASK WITH THIS EDGE DOWN
        //MASKTYPE NewMask = new unsigned char[MaskSize]; 
        for (int i = 0; i < MaskSize; i++) NewMask[i] = Mask[i];
        NewMULT = MULT;
        NewMask[sn] = 0;
        NewMULT = NewMULT * (1 - Prob[sn]);
        //for (int i = 0; i < MaskSize; i++) printf("%2x**",NewMask[i]); printf("\n");
        if (Opermask(N, NewMask, Nodes, Edges, src, t, Visited[tx], Parent[tx], Queue[tx])) { // DOES NOT HAVE A CUT
          //printf ("OPER\n");
          if (Q_SIZE < MAX_Q_SIZE) {
            MQXRear++;
            for (int i = 0; i < MaskSize; i++) MASKQUE[MQXRear][i] = NewMask[i];
            MKEY[MQXRear] = NewMULT;
            Q_SIZE++;
            // HEAPIFY(MASKQUE, MKEY, TempM, Q_SIZE);   

          } else { // MAX QUEUE SIZE EXCEEDED

            //printf("MAX QUEUE SIZE EXCEEDED\n");   // --> Rel_10 // DON'T ADD NewMask TO QUEUE & Seq_10.cpp
            // add its prob with 1's replacing-1s (don't cares) else do nothing)
            NP[tx]++;
            if (isnan(NewMULT)) printf("ISNAN\n");
            if (!isnan(NewMULT)) {
              Epsilon[tx] += NewMULT;
              //printf("TID: %d NP = %d NewMULT  %.20lf Eps %.20lf +++\n", TID, NP[TID], NewMULT, Eps[TID]);
            }
          } //  else
        } // if Opermask (New Musk)
        Mask[sn] = 1;
        MULT *= Prob[sn]; // MARK THIS EDGE UP IN THE CURRENT MASK           	      
      } // while	     	     
      // Add Increment to rel[tx],
      NP[tx]++;
      //for (int i = 0; i < MaskSize; i++) printf("%2x*",Mask[i]); printf("\n");
      rel[tx] += MULT; //if (TID == 0) printf("NP = %d MULT  %.20lf rel %.20lf +++\n", NP[TID], MULT, rel[TID]);
      //mTime = (int) (((float) clock() - start)/  CLOCKS_PER_SEC) ; // clock() measures the total time of all threads??
      // 	if (mTime > 112400) {Epsilon[tx] += PURGEQ(MQXFront, MQXRear, MKEY);if (TID == 0) printf("PURGE\n");
      //	  break; 	}     	                              
    } // if (OP)		
  } // END OF MASK QUEUE PROCESSING
  /*____________________________________________________________*/

  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
    __syncthreads();
    if (tx < stride) {
      rel[tx] += rel[tx + stride];
      NP[tx] += NP[tx + stride];
      Epsilon[tx] += Epsilon[tx + stride];
    }
  }
  __syncthreads();  // ADD: Wait after final iteration
  
  //__syncthreads(); 
  if (tx == 0) {
    d_Brel[bx] = rel[0];
    d_NP[bx] = NP[0];
    d_Epsilon[bx] = Epsilon[0];
  }
  //__syncthreads();
  //printf("RELA = %lf ", d_Brel[0]);
  for (int i = 0; i < MAX_Q_SIZE; i++) {
    delete[] MASKQUE[i];
  }
  delete[] MASKQUE;
  delete[] Mask;
  delete[] NewMask;
  delete[] MKEY;
  return;
}

/*______________________________________________________________________________*/

int main() {
  //printf("x = %lf ", 1- pow(0.75, 15));

  size_t free, total;
  cudaMemGetInfo( & free, & total);
  printf("TOTAL %lu  FREE %lu \n", total, free);

  int deviceCount;
  cudaGetDeviceCount( & deviceCount);
  cout << " GPU Count : " << deviceCount << endl;

  size_t * H = new(size_t);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 3072 * 1024 * 1024LL);
  cudaDeviceGetLimit(H, cudaLimitMallocHeapSize);
  printf(" HEAP %lu \n", * H);

  //bool SMALL =false; int ET;int TNTH = NBL *NTH;// Total # threads for big graphs
  Graph_Type * G1 = new Graph_Type; // INPUT GRAPH
  Edge_Type * e_l;
  //cout << sizeof(Edge_Type);
  string fname;
  clock_t start, end;
  cout << " Enter Input File Name: ";
  cin >> fname;
  cout << " Enter src & terminal: ";
  cin >> src >> t;
  cout << "Enter NB : ";
  cin >> NBL;
  infile.open(fname.c_str());
  if (!infile) std::cout << "Input File Error!!\n";
  start = clock();
  ReadGraph(G1);
  N = G1 -> N;
  M = G1 -> M;
  cout << "READ OK\n";
  N_Type * Nodes = new N_Type[N];
  E_Type * Edges = new E_Type[M];
  Nodes[0].Offset = 0;
  int E_X = 0;
  for (int n = 0; n < N; n++) {
    int d = G1 -> Nodes[n].degree;
    Nodes[n].degree = d;
    if (n > 0) Nodes[n].Offset = Nodes[n - 1].Offset + Nodes[n - 1].degree;
    e_l = G1 -> Nodes[n].Adj_List;
    for (int dx = 0; dx < d; dx++) {
      Edges[E_X].SeqNo = e_l[dx].SeqNo;
      Edges[E_X].prob = e_l[dx].prob;
      Edges[E_X].dst = e_l[dx].dst;
      E_X++;
    }
  }
  //PrintGraph (G1, Nodes, Edges);
  //PrintSeqNos(N, M , Nodes, Edges);
  //printf(" SSSS = %d ",Seq_No( 1, 2, Nodes, Edges)); 
  int NumOfEdges = M;
  if (G1 -> GT == 'U') NumOfEdges = M / 2;
  Prob = new double[NumOfEdges]; // Enough precision

  for (int i = 0; i < M; i++) {
    int sn = Edges[i].SeqNo; //if (sn >= M/2) cout << "ERROR"; 
    Prob[sn] = Edges[i].prob;
  }

  // In main(), after Prob is populated:
  for (int i = 0; i < NumOfEdges; i++) {
    if (Prob[i] < 0.0 || Prob[i] > 1.0) {
      printf("Error: Invalid probability Prob[%d] = %f\n", i, Prob[i]);
      return 1;
    }
  }

  for (int i = 0; i < NGpus; i++) {
    cudaSetDevice(i);

    cudaMalloc( & d_Graph[i], sizeof(Graph_Type) *NBL );
    cudaMalloc( & d_Brel[i], sizeof(double) * NBL);
    cudaMalloc( & d_BEps[i], sizeof(double) * NBL);
    cudaMalloc( & d_BNP[i], sizeof(int) * NBL);
    cudaMalloc( & d_Nodes[i], sizeof(N_Type) * N);
    cudaMalloc( & d_Edges[i], sizeof(E_Type) * M);
    cudaMalloc( & d_Prob[i], sizeof(double) * M);
    cudaMemcpyAsync(d_Graph[i], G1, sizeof(Graph_Type), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_Nodes[i], Nodes, sizeof(N_Type) * N, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_Edges[i], Edges, sizeof(E_Type) * M, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_Prob[i], Prob, sizeof(double) * M, cudaMemcpyHostToDevice);
    cudaMemset(d_Brel[i], 0, sizeof(double) * NBL);
    cudaMemset(d_BEps[i], 0, sizeof(double) * NBL);
    cudaMemset(d_BNP[i], 0, sizeof(int) * NBL); 
  } // End Alloc & CPY

  int TID_SIZE = log2(NBL * NTH * NGpus); // cannot be called from kernel
  for (int i = 0; i < NGpus; i++) {
    cudaSetDevice(i);
    TC << < NBL, NTH >>> (d_Graph[i], d_Nodes[i], d_Edges[i], d_Prob[i], src, t, d_Brel[i], d_BEps[i], d_BNP[i], TID_SIZE, i);
  }

  for (int i = 0; i < NGpus; i++) {
    cudaSetDevice(i);
    {
      cudaError_t cudaerr = cudaDeviceSynchronize();
      if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
          cudaGetErrorString(cudaerr));
    }
  }
  for (int i = 0; i < NGpus; i++) { // HOST DATA NEEDS [] Too ?????????????????????????????????
    cudaMemcpyAsync(Brel[i], d_Brel[i], sizeof(double) * NBL, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(BNP[i], d_BNP[i], sizeof(int) * NBL, cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(BEps[i], d_BEps[i], sizeof(double) * NBL, cudaMemcpyDeviceToHost);
  }
  double h_rel = 0.0, rel[NGpus] = {
    0.0
  }, h_Eps = 0.0, Eps[NGpus] = {
    0.0
  };
  int h_NP = 0, NP[NGpus] = {
    0
  };
  cout << "BACK ";

  for (int d = 0; d < NGpus; d++) {
    for (int i = 0; i < NBL; i++) {
      rel[d] += Brel[d][i];

    }
    printf(" rel[ %d] (Lower Bound) = %.10lf \n", d, rel[d]);

    for (int i = 0; i < NBL; i++) {
      Eps[d] += BEps[d][i];
    }
    printf(" Eps[%d] = %.20lf Upper Bound = %.20lf \n", d, Eps[d], rel[d] + Eps[d]);
    for (int i = 0; i < NBL; i++)
      NP[d] += BNP[d][i];
    printf(" NP[ %d] = %d \n", d, NP[d]);
  }

  end = clock();
  cout << "Time = " << ((double) end - start) * 1000 / CLOCKS_PER_SEC << " ms \n";

  for (int d = 0; d < NGpus; d++) {
    h_rel += rel[d];
    h_Eps += Eps[d];
    h_NP += NP[d];
  }
  printf(" Totals: %.20lf  %.20lf  %d \n", h_rel, h_Eps, h_NP);

  for (int i = 0; i < NGpus; i++) {
    cudaFree((Graph_Type * ) d_Graph[i]);
    cudaFree((double * ) d_Brel[i]);
    cudaFree((double * ) d_Prob[i]);
    cudaFree((int * ) d_BNP[i]);
    cudaFree((double * ) d_BEps[i]);
    cudaFree((Node_Type * ) d_Nodes[i]);
    cudaFree((Edge_Type * ) d_Edges[i]);
  }

  return 0;
}