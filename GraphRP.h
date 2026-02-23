
void Adjust_SeqNos(Graph_Type *G, int N) {
 Edge_Type * e1,  * e2; short dst, SN; int d, offset;
 offset = G->Nodes[0].degree;
 for (int i = 1; i < N; i++) {
    d = G->Nodes[i].degree; 
    e1 = G->Nodes[i].Adj_List;
    for (int j= 0; j < d; j++) {
     dst = e1[j].dst; // Edge (i, j)
     if (dst < i) {  // j < i 
       e2 = G->Nodes[dst].Adj_List; int k = 0;
       while ( e2[k].dst != i) k++;
       SN = e2[k].SeqNo; e1[j].SeqNo = SN;
      } else { e1[j].SeqNo = offset ; offset++; }
     } // for j 
   } // for i
  return;
}  
	
	
void ReadGraph (Graph_Type *G ) {
	short Nnodes, M; char Gtype;
	infile >> Nnodes >> M >> Gtype; 
	cout << "N= " << Nnodes << " M = " << M << " type: "  << Gtype << endl;
	G->N = Nnodes;G->M = M; G->GT = Gtype;
	//printf("HERE 1");
	int degree; Edge_Type *e;
	G-> Nodes = new Node_Type[Nnodes];// G->Nodes = (Node_Type *) malloc(N* sizeof(Node_Type));
	short SeqNo = 0; 
	//printf("Here 2");
	for (int i = 0; i < Nnodes; i++) {
	  infile >> degree; //cout << " DEGREE " << degree;
	  G->Nodes[i].degree = degree; 
	  if (degree > 0) G->Nodes[i].Adj_List =  new Edge_Type[degree];
	  e = G->Nodes[i].Adj_List; 
	   for (int j =0; j < degree ; j++)
	      { 
	        infile >> e[j].dst; infile >> e[j].prob; e[j].SeqNo = SeqNo++;
	       }
	}
       if (Gtype == 'U') Adjust_SeqNos(G, Nnodes);
       return;
}

        

void PrintGraph (Graph_Type *G, N_Type Nodes[], E_Type Edges[]) {
	Edge_Type *e; int d;
	cout << "N: " << G->N << "M: " << G->M << G->GT << endl;
	int N = G->N;int M = G->M;
	for (int i = 0 ; i < N; i++ ) {
		e = G->Nodes[i].Adj_List; 
		d = G->Nodes[i].degree; 
		cout << i << " :: " << d << ":: ";
		for (int j =0; j < d; j++)  {
		    printf (" PROB %.3lf ", e[j].prob);
		    cout << e[j].dst << ": " << e[j].prob << " SN= "<< e[j].SeqNo << ": ";}
		cout << endl;
	 }  
	 printf("\n Degrees: ");for (int i = 0; i < N; i++) printf(" %d ", Nodes[i].degree); 
	 printf("\n Offsets: ");for (int i = 0; i < N; i++) printf(" %d ", Nodes[i].Offset);
	 printf("\n Edges:  "); for (int i=0; i<M; i++)  printf(" %d %d %f :", Edges[i].SeqNo, Edges[i].dst, Edges[i].prob);
	 printf ("\n*********************\n");
	return;
}


