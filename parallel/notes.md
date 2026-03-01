
## Function `TID_TO_MASK` Behavior Based on `TID_SIZE` and `MASK_SIZE`

The function behavior depends on the relationship between `TID_SIZE` and `MASK_SIZE`.

- The first `TID_SIZE` edges are assigned a definite state (`up` or `down`), and the remaining edges are marked as undecided (`0xFF`).
- Let the decided portion be denoted as **A**.
  - In **A**, not all possible combinations of `up` and `down` are generated unless `|A| = 2^MASK_SIZE`.
  - If **A** results in a disconnected graph, no further processing can be performed at this stage.
  - If `TID_SIZE >= MASK_SIZE and TID_SIZE < 2^MASK_SIZE`, then no edge remains marked as `0xFF`, and the factoring algorithm cannot proceed.

- If `TID_SIZE = 2^MASK_SIZE`, all possible edge-state configurations are generated exactly once.
- If `TID_SIZE > 2^MASK_SIZE`, some configurations are generated more than once (duplication occurs).
```c++
// This is an example to see generated configurations. 

#include <iostream>
using namespace std;
int main(int argc, char *argv[]) {
	int MaskSize = 8; 
	int TID_SIZE = 4; 
	char* mask = new char[MaskSize]; 
	
	for (int e = 0; e < MaskSize; e++) 
		mask[e] = '-'; // UNMARKED EDGE
	for(int tid = 0; tid < TID_SIZE; tid++){
		if (TID_SIZE <= MaskSize) 
			for (int e = 0; e < TID_SIZE; e++) { 
				int m = 1 << e; 
				if ( tid & m) 
					mask[e] = 'x'; 
				else
					mask[e] = 't';
			}
		else
			for (int e = 0; e < MaskSize; e++) {
				int m = 1 << e; 
				if ( tid & m) 
					mask[e] = 'x'; 
				else 
					mask[e] = 't';
			}
		for(int i = 0; i < MaskSize; i++){
			cout<<mask[i]; 
		}
		cout<<endl;
	}
}
/*
output
tttt----
xttt----
txtt----
xxtt----
*/
```
