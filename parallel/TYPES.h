
typedef enum {Up = 1, Down = 0, NoMark = -1} edgeState;
typedef struct { float prob; short SeqNo; short dst; } Edge_Type;
typedef struct { short degree; Edge_Type *Adj_List; } Node_Type;
typedef struct { short N; short M; Node_Type *Nodes; unsigned char GT;} Graph_Type;
typedef struct { float prob; short SeqNo; short dst; } E_Type;
typedef struct { short degree; short Offset; } N_Type;






