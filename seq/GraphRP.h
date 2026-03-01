#ifndef GRAPHRP_H
#define GRAPHRP_H

//=============================================================================
// GraphRP.h - Graph Reading and Printing Functions
// Improved version with better error handling and documentation
//=============================================================================

#include <iostream>
#include <fstream>
#include <cstdio>

using namespace std;

extern ifstream infile;  // Declared in main file

//-----------------------------------------------------------------------------
// Adjust_SeqNos: For undirected graphs, ensure both (i,j) and (j,i) share SeqNo
// This reduces the mask size by half for undirected graphs
//-----------------------------------------------------------------------------
void Adjust_SeqNos(Graph_Type *G, int N) {
    Edge_Type *e1, *e2;
    EdgeId SN;
    NodeId dst;
    EdgeId offset;
    
    offset = G->Nodes[0].degree;
    
    for (int i = 1; i < N; i++) {
        int d = G->Nodes[i].degree;
        e1 = G->Nodes[i].Adj_List;
        
        for (int j = 0; j < d; j++) {
            dst = e1[j].dst;  // Edge (i, dst)
            
            if (dst < i) {
                // This edge was already seen from the other direction (dst, i)
                // Find edge (dst, i) and copy its SeqNo
                e2 = G->Nodes[dst].Adj_List;
                int k = 0;
                while (e2[k].dst != i) k++;
                SN = e2[k].SeqNo;
                e1[j].SeqNo = SN;
            } else {
                // New edge, assign next SeqNo
                e1[j].SeqNo = offset;
                offset++;
            }
        }
    }
}

//-----------------------------------------------------------------------------
// printG: Debug function to print graph structure
//-----------------------------------------------------------------------------
void printG(Graph_Type *G, int N) {
    for (int i = 0; i < N; i++) {
        Edge_Type *e1 = G->Nodes[i].Adj_List;
        int m = G->Nodes[i].degree;
        printf("%d: ", i);
        for (int j = 0; j < m; j++) {
            printf("(%d, SN=%d, p=%.3f) ", e1[j].dst, e1[j].SeqNo, e1[j].prob);
        }
        printf("\n");
    }
}

//-----------------------------------------------------------------------------
// ReadGraph: Read graph from input file
// 
// File format:
//   N M Type
//   degree_0 dst1 prob1 dst2 prob2 ...
//   degree_1 dst1 prob1 dst2 prob2 ...
//   ...
//
// Where:
//   N = number of nodes
//   M = number of edge entries (for undirected: 2 * actual edges)
//   Type = 'U' for undirected, 'D' for directed
//-----------------------------------------------------------------------------
bool ReadGraph(Graph_Type *G) {
    NodeId Nnodes;
    EdgeId M;
    char Gtype;
    
    // Read header
    infile >> Nnodes >> M >> Gtype;
    if (infile.fail()) {
        cerr << "Error reading graph header" << endl;
        return false;
    }
    
    cout << "N = " << Nnodes << ", M = " << M << ", Type: " << Gtype << endl;
    
    G->N = Nnodes;
    G->M = M;
    G->GT = Gtype;
    
    // Allocate nodes
    G->Nodes = new Node_Type[Nnodes];
    if (!G->Nodes) {
        cerr << "Failed to allocate nodes array" << endl;
        return false;
    }
    
    EdgeId SeqNo = 0;
    
    // Read each node's adjacency list
    for (int i = 0; i < Nnodes; i++) {
        int degree;
        infile >> degree;
        if (infile.fail()) {
            cerr << "Error reading degree for node " << i << endl;
            return false;
        }
        
        G->Nodes[i].degree = degree;
        
        if (degree > 0) {
            G->Nodes[i].Adj_List = new Edge_Type[degree];
            if (!G->Nodes[i].Adj_List) {
                cerr << "Failed to allocate adjacency list for node " << i << endl;
                return false;
            }
        } else {
            G->Nodes[i].Adj_List = nullptr;
        }
        
        Edge_Type *e = G->Nodes[i].Adj_List;
        
        for (int j = 0; j < degree; j++) {
            infile >> e[j].dst >> e[j].prob;
            if (infile.fail()) {
                cerr << "Error reading edge " << j << " for node " << i << endl;
                return false;
            }
            e[j].SeqNo = SeqNo++;
        }
    }
    
    // For undirected graphs, make edge pairs share SeqNo
    if (Gtype == 'U') {
        Adjust_SeqNos(G, Nnodes);
    }
    
    return true;
}

//-----------------------------------------------------------------------------
// PrintGraph: Print complete graph information for debugging
//-----------------------------------------------------------------------------
void PrintGraph(Graph_Type *G, N_Type Nodes[], E_Type Edges[]) {
    Edge_Type *e;
    int d;
    
    cout << "\n=== Graph Structure ===" << endl;
    cout << "N: " << G->N << ", M: " << G->M << ", Type: " << (char)G->GT << endl;
    
    int N = G->N;
    int M = G->M;
    
    cout << "\nAdjacency List:" << endl;
    for (int i = 0; i < N; i++) {
        e = G->Nodes[i].Adj_List;
        d = G->Nodes[i].degree;
        cout << "  Node " << i << " (degree=" << d << "): ";
        for (int j = 0; j < d; j++) {
            cout << "(" << e[j].dst << ", p=" << e[j].prob << ", SN=" << e[j].SeqNo << ") ";
        }
        cout << endl;
    }
    
    cout << "\nCompact Representation:" << endl;
    cout << "  Degrees: ";
    for (int i = 0; i < N; i++) cout << Nodes[i].degree << " ";
    cout << endl;
    
    cout << "  Offsets: ";
    for (int i = 0; i < N; i++) cout << Nodes[i].Offset << " ";
    cout << endl;
    
    cout << "  Edges: ";
    for (int i = 0; i < M; i++) {
        cout << "[SN=" << Edges[i].SeqNo << ", dst=" << Edges[i].dst 
             << ", p=" << Edges[i].prob << "] ";
    }
    cout << endl;
    cout << "========================\n" << endl;
}

//-----------------------------------------------------------------------------
// FreeGraph: Clean up graph memory
//-----------------------------------------------------------------------------
void FreeGraph(Graph_Type *G) {
    if (G->Nodes) {
        for (int i = 0; i < G->N; i++) {
            if (G->Nodes[i].Adj_List) {
                delete[] G->Nodes[i].Adj_List;
            }
        }
        delete[] G->Nodes;
    }
}

#endif // GRAPHRP_H