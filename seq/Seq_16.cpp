// From Rel_10  2024 -> Seq_10 (Dynamic) -> Seq_11 : USES FIXED MEMORY FOR MASKQUE 27/1/24
//From Seq_12.cpp 16/2/2024 : Add Mask to Queue only if Operational (No Cut exists)
// HEAP (key is MULT)
#include "TYPES.h"

#include <stdio.h>

#include <assert.h>

#include <cstdlib>

#include <cstring>

#include <cmath>

#include <ctime>

#include <set>

#include <iostream>

#include <fstream>

#include <string>

#include <sys/resource.h>

#include <new>

using namespace std;
ifstream infile;
#include "GraphRP.h"

#define MAX_Q_SIZE 1300000
typedef unsigned char * MASKTYPE;
double MKEY[MAX_Q_SIZE];
MASKTYPE * MASKQUE = new MASKTYPE[MAX_Q_SIZE]; // MAXIMUM NUMBER OF MASKS ?????
int MQXFront = 0, MQXRear = -1; // FRONT is the root of the HEAP
int Q_SIZE;
int NP = 0;
double rel = 0.0, Eps = 0.0;
int N; //number of nodes
int M; //number of edges
unsigned char GT;
int src, t; //  source node & destination node
set < int > target_set; // used for K-terminal Reliability

bool * Visited;
short * Parent;
short * Queue;
MASKTYPE TempM;
double TempK;
void CopyMask(int from, int to, int MS) {
  int i;
  for (int i = 0; i < MS; i++) MASKQUE[to][i] = MASKQUE[from][i];
  MKEY[to] = MKEY[from];
  return;
}
double PURGEQ() {
  double SumQ = 0;
  for (int i = MQXFront; i <= MQXRear; i++) SumQ += MKEY[i];
  return SumQ;
}
void QDUMP(int MS) {
  char OK;
  int i = MQXFront, j = MQXRear, QS = j - i + 1;;
  printf(" QS = %d\n", QS);
  for (int x = i; x <= j; x++) {
    printf(" MASKQUE[%d] %p : ", x, MASKQUE[x]);
    for (int k = 0; k < MS; k++) printf(" %x ", MASKQUE[x][k]);
    printf(" MKEY = %lf \n", MKEY[x]);
  }
  //cin >> OK;
}

void __inline__ REHEAPIFY(int n) { // From root down
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

void __inline__ HEAPIFY(int n) { // From down up
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
bool CHECKMULT(float Prob[], int MaskSize, int IX) {
  double XM = 1.0;
  for (int i = 0; i < MaskSize; i++) {
    if (MASKQUE[IX][i] == 1) XM *= Prob[i];
    if (MASKQUE[IX][i] == 0) XM *= (1 - Prob[i]);
  }
  return (XM == MKEY[IX]);
}
int Seq_No(int i, int j, N_Type Nodes[], E_Type Edges[]) {
  int offset = Nodes[i].Offset;
  int degree = Nodes[i].degree;
  for (int d = 0; d < degree; d++) {
    E_Type e = Edges[offset + d];
    if (e.dst == j) return (e.SeqNo);
  }
  return 0;
}
void PrintSeqNos(int N, int M, N_Type Nodes[], E_Type Edges[]) {
  printf("\n Degrees: ");
  for (int i = 0; i < N; i++) printf(" %d ", Nodes[i].degree);
  printf("\n Offsets: ");
  for (int i = 0; i < N; i++) printf(" %d ", Nodes[i].Offset);
  printf("\n Edges:  ");
  for (int i = 0; i < M; i++) printf(" %d %d %f :", Edges[i].SeqNo, Edges[i].dst, Edges[i].prob);
  printf("\n*********************\n");
}
bool Opermask(int N, unsigned char * NewMask, N_Type Nodes[], E_Type Edges[], int src, int t) {
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
      int sn = Edges[e_l + dx].SeqNo; //printf("0%xHERE %d %d %d|", V, j, sn);
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

void TC(Graph_Type * G, N_Type Nodes[], E_Type Edges[], float Prob[], int src, int t) {
  int M = G -> M, N = G -> N;
  unsigned char GT = G -> GT;
  int sn;
  clock_t start = clock(), end;
  int mTime;
  bool OP = false;
  double MULT, NewMULT;
  int e_l;
  int d, dx;
  short V, j;;
  int MaskSize;
  for (int i = 0; i < MAX_Q_SIZE; i++) MKEY[i] = 1.0;
  if (GT == 'U') MaskSize = M / 2;
  else MaskSize = M;

  if (MASKQUE) printf("OK %p \n", MASKQUE);
  else printf("Memory Problem!!");
  for (int i = 0; i < MAX_Q_SIZE; i++) {
    MASKQUE[i] = new unsigned char[MaskSize];
    if (!MASKQUE[i]) {
      printf("NULL POINTER");
      break;
    }
  }
  printf("PASSED MEMORY \n");
  MASKTYPE Mask = new unsigned char[MaskSize], NewMask = new unsigned char[MaskSize]; // WORKING MASKs
  printf(" MaskSize = %d MAX_Q_SIZE = %d\n", MaskSize, MAX_Q_SIZE);

  for (int i = 0; i < MaskSize; i++) Mask[i] = 0xFF; // INITIAL MASK : ALL -1
  short * TQ = new short[MaskSize];
  int TQTOP; // STACK TO store sn for unmarked edges ON AN s-t-path -> Rel_08.cu

  bool * Visited = new bool[N];
  short * Parent = new short[N];
  short * Queue = new short[N]; // BFS src -> t

  /*________________START PROCESSING MASK QUEUE ________________________________*/

  MQXRear++;
  for (int i = 0; i < MaskSize; i++) MASKQUE[MQXRear][i] = Mask[i];
  MKEY[MQXRear] = 1.0;
  Q_SIZE = 1; //for (int i = 0; i < MaskSize; i++) printf("0x%.2x  ",MASKQUE[0][i]); printf("\n");
  int T = 0;
  int TT = 0;
  while (Q_SIZE > 0) {
    TT++; //printf("BEFORE:: ");QDUMP(MaskSize);
    check: //for (int i =0; i < 10; i++) printf(" %p : ",MASKQUE[i]); printf("SIZE = %d \n", Q_SIZE);
      //printf ("MQXFront = %d T = %d ", MQXFront, T);
      for (int i = 0; i < MaskSize; i++) Mask[i] = MASKQUE[MQXFront][i];
    MULT = MKEY[MQXFront]; //if (!CHECKMULT(Prob, MaskSize, MQXFront)) printf("ERROR1: IX = %d T= %d \n", MQXFront, T);
    if (Q_SIZE > 1) {
      CopyMask(MQXRear, MQXFront, MaskSize);
      MQXRear--;
    }
    Q_SIZE -= 1;
    if (Q_SIZE == 0) MQXRear = -1;

    //printf("BEFORE:: ");QDUMP(MaskSize);
    REHEAPIFY(Q_SIZE); //printf("AFTER:: ");QDUMP(MaskSize);

    //printf("AFTER::  ");QDUMP(MaskSize);
    //printf( " T = %d Q_SIZE(AFTER) = %d MULT %lf", T, Q_SIZE, MULT); for (int i = 0; i < MaskSize; i++) printf(" %.2x ",Mask[i]); printf("\n");
    T++;
    // GET THE s,t-path in this Mask
    for (int i = 0; i < N; i++) Visited[i] = false;
    OP = false;
    for (int i = 0; i < N; i++) Parent[i] = -1;

    int front = 0, rear = -1;
    rear++;
    Queue[rear] = (short) src;
    while (rear >= front) {
      V = Queue[front];
      front++; //////// //printf("V = %d|", V);
      d = Nodes[V].degree;
      e_l = Nodes[V].Offset;
      for (dx = 0; dx < d; dx++) {
        j = Edges[e_l + dx].dst;
        sn = Edges[e_l + dx].SeqNo; //printf("HERE %d %d %d|", V, j, sn);
        if (Mask[sn] == 1 || Mask[sn] == 0xFF)
          if (!Visited[j]) {
            Visited[j] = true;
            rear++;
            Queue[rear] = j;
            Parent[j] = V;
          }

      }
      if (Visited[t]) {
        OP = true;
        break;
      }
    }

    //printf("OP %d  ", OP);
    //for (int i = 0; i < MaskSize; i++) printf("%.2x ",Mask[i]); //printf("\n");

    if (OP) {
      // Climb up the path from t to s; Store sn's for unmarked edges
      int i = Parent[t];
      int j = t;
      int sn;
      TQTOP = -1;
      while (j != src) {
        sn = Seq_No(i, j, Nodes, Edges);
        if (Mask[sn] == 0xFF) {
          TQTOP++;
          TQ[TQTOP] = sn; //cout << "(" << i << j << sn << ")" ;
        }
        j = i;
        i = Parent[j];
      }

      while (TQTOP >= 0) {
        sn = TQ[TQTOP];
        TQTOP--; //Create EXTRA MASKS from TQ
        // cREATE NEW MASK 
        for (int i = 0; i < MaskSize; i++) NewMask[i] = Mask[i];
        NewMULT = MULT;
        NewMask[sn] = 0;
        NewMULT = NewMULT * (1 - Prob[sn]); //printf(" Q SIZE = %d\n", Q_SIZE);
        if (Opermask(N, NewMask, Nodes, Edges, src, t)) { // DOES NOT HAVE A CUT
          if (Q_SIZE < MAX_Q_SIZE) { //printf("HEER %d\n", MQXRear);
            MQXRear++;
            Q_SIZE++;
            for (int i = 0; i < MaskSize; i++) MASKQUE[MQXRear][i] = NewMask[i];
            MKEY[MQXRear] = NewMULT;
            //printf("BEFORE-:: ");QDUMP(MaskSize);
            HEAPIFY(Q_SIZE); //printf("AFTER:: ");QDUMP(MaskSize);
            //  if (!CHECKMULT(Prob, MaskSize, MQXRear)) printf("ERROR\n");else printf("OK  \n");
            //printf("** MQXRear = %d MQXFront = %d Q_SIZE = %d\n", MQXRear, MQXFront, Q_SIZE);
            //for (int i = 0; i < MaskSize; i++) printf(" %.2x &",MASKQUE[MQXRear][i]); printf (" MKEY %lf ",MKEY[MQXRear]); printf(" ADDED\n");  
            //for (int i = 0; i < MaskSize; i++) printf(" %.2x &",MASKQUE[MQXFront][i]); printf("\n");  //goto check;
          } else {
            //printf("MAX QUEUE SIZE EXCEEDED\n");   // --> Rel_10 // DON'T ADD NewMask TO QUEUE & Seq_10.cpp
            // add its prob with 1's replacing-1s (don't cares) else do nothing)}
            NP++;
            if (isnan(NewMULT)) printf("ISNAN\n");
            if (!isnan(NewMULT)) {
              Eps += NewMULT;
              //printf("NP = %d NewMULT  %.20lf Eps %.20lf +++\n", NP, NewMULT, Eps);
            }

          } // else
        } // if OPER
        Mask[sn] = 1;
        MULT *= Prob[sn]; // MARK THIS EDGE UP IN THE CURRENT MASK
        //printf (" SN = %d MULT = %lf   Prob[sn] %f ", sn , MULT, Prob[sn]);
      } // while       

      // Add Increment to rel[tx],
      NP++;
      if (isnan(MULT)) printf("ISNAN\n");
      if (!isnan(MULT)) {
        rel += MULT; //printf("NP = %d MULT  %.20lf rel %.20lf +++\n", NP, MULT, rel);
      }
      mTime = (int)(((float) clock() - start) / CLOCKS_PER_SEC);
      // if (mTime > 120) {Eps += PURGEQ();break; 	}
    } // if(OP)                           
    end = clock();
    mTime = (int)(((float) end - start) / CLOCKS_PER_SEC);
    /* if ( mTime % 100 == 0)	//return;
     cout << mTime << " : " << rel << " : " << Eps << " : " << Q_SIZE <<endl;*/
  } // END OF MASK QUEUE PROCESSING
  /*____________________________________________________________*/
  cout << "TT = " << TT << endl;

  return;
}
/*______________________________________________________________________________*/

float * Prob;

struct rusage usage;

int main() {

  Graph_Type * G1 = new Graph_Type; // INPUT GRAPH
  Edge_Type * e_l;

  string fname;
  clock_t start, end;
  cout << " Enter Input File Name: ";
  cin >> fname;
  cout << " Enter src & terminal: ";
  cin >> src >> t;
  infile.open(fname.c_str());
  if (!infile) std::cout << "Input File Error!!\n";
  start = clock();
  ReadGraph(G1);
  N = G1 -> N;
  M = G1 -> M;
  cout << "READ OK\n";
  Visited = new bool[N];
  if (!Visited) printf("OUT OF MEMORY");
  Parent = new short[N];
  if (!Parent) printf("OUT OF MEMORY");
  Queue = new short[N];
  if (!Queue) printf("OUT OF MEMORY");
  N_Type * Nodes = new N_Type[N];
  E_Type * Edges = new E_Type[M];
  if (!Edges) printf("OUT OF MEMORY");
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
  Prob = new float[NumOfEdges];
  if (!Prob) printf("OUT OF MEMORY");

  for (int i = 0; i < M; i++) {
    int sn = Edges[i].SeqNo;
    Prob[sn] = Edges[i].prob;
    //printf(" SN %d PROB %f ", sn, Prob[sn]);
  }

  TC(G1, Nodes, Edges, Prob, src, t);
  getrusage(RUSAGE_SELF, & usage);
  cout << "Max RSS (heap size): " << usage.ru_maxrss << " kilobytes" << std::endl;

  printf(" rel = %.20lf \n", rel);
  printf(" Eps = %.20lf rel+Eps %.20lf\n", Eps, rel + Eps);

  cout << " NP = " << NP << endl;
  end = clock();
  cout << "Time = " << ((float) end - start) * 1000 / CLOCKS_PER_SEC << " ms \n";

  return 0;
}