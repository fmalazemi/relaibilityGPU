#ifndef TYPES_H
#define TYPES_H

//=============================================================================
// TYPES.h - Data type definitions for Network Reliability
// Improved version with consistent types and better documentation
//=============================================================================

#include <cstdint>

//-----------------------------------------------------------------------------
// Type Aliases for Clarity and Consistency
//-----------------------------------------------------------------------------
typedef double Probability;    // Use double consistently for probabilities
typedef int32_t NodeId;        // Supports up to 2 billion nodes
typedef int32_t EdgeId;        // Supports up to 2 billion edges

//-----------------------------------------------------------------------------
// Edge State Enumeration
//-----------------------------------------------------------------------------
typedef enum { 
    Up = 1,       // Edge is operational (working)
    Down = 0,     // Edge has failed
    NoMark = -1   // Edge state not yet determined (don't care)
} EdgeState;

// Mask values (unsigned char representation)
#define MASK_UP     0x01    // Edge is UP
#define MASK_DOWN   0x00    // Edge is DOWN  
#define MASK_NOMARK 0xFF    // Edge is unmarked (don't care)

//-----------------------------------------------------------------------------
// Edge Structure for Adjacency List Representation
//-----------------------------------------------------------------------------
typedef struct { 
    Probability prob;   // Edge reliability probability (0 < prob < 1)
    EdgeId SeqNo;       // Unique edge identifier
    NodeId dst;         // Destination node
} Edge_Type;

//-----------------------------------------------------------------------------
// Node Structure for Adjacency List
//-----------------------------------------------------------------------------
typedef struct { 
    NodeId degree;          // Number of adjacent edges
    Edge_Type *Adj_List;    // Array of adjacent edges
} Node_Type;

//-----------------------------------------------------------------------------
// Main Graph Structure
//-----------------------------------------------------------------------------
typedef struct { 
    NodeId N;               // Number of nodes
    EdgeId M;               // Number of edges (undirected: counts each edge twice)
    Node_Type *Nodes;       // Array of nodes
    unsigned char GT;       // Graph type: 'U' = undirected, 'D' = directed
} Graph_Type;

//-----------------------------------------------------------------------------
// Compact Edge Representation (CSR-like format for main algorithm)
//-----------------------------------------------------------------------------
typedef struct { 
    Probability prob;   // Edge probability
    EdgeId SeqNo;       // Edge sequence number
    NodeId dst;         // Destination node
} E_Type;

//-----------------------------------------------------------------------------
// Compact Node Representation (CSR-like format)
//-----------------------------------------------------------------------------
typedef struct { 
    NodeId degree;      // Node degree
    EdgeId Offset;      // Starting index in the Edges array
} N_Type;

#endif // TYPES_H