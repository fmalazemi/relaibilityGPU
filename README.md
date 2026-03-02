## Notes
1. If we use a heap for masks. If the heap is full and we extracted most significant mask X, then we can only store one branch from X even if the path from src to dst includes Y number of Unkown edges.
```

------- COMP10-05.txt --------------- 



============================================
Two-Terminal Network Reliability Results (NEW)
============================================
Graph:            N = 10, E = 45
Terminals:        src = 0, dst = 9
--------------------------------------------
Reliability:      R(G) = 0.973826173985
R + ESP:                 0.973826173985
ESP (truncated):         0.000000e+00
Truncation bound:        0.000000e+00
--------------------------------------------
Paths enumerated: 163622734
Nodes processed:  1134455529
Kernel iterations:46
Max queue RSS:    6620.576 MB  (86777216 items)
Total time:       78894.266 ms
============================================



=== Starting CUDA Reliability Computation === (OLD)
Nodes: 10, Edges: 90, MaskSize: 45
Batch size: 1024

=== Computation Complete ===
Total iterations: 7433032
Paths processed:  7433032
Time:             9.42317 seconds
Rate:             788804 iter/sec

=== RESULTS ===
Reliability:       0.99596245255088433623
Truncation (Eps):  0.00000000000000000000
Total:             0.99596245255088433623
Paths enumerated:  7433032
Total time:        9573.85 ms
Max RSS:           202528 KB


============= Original =============== 
TOTAL 47709159424  FREE 47427092480 
GPU Count : 1
HEAP 3221225472 
Enter Input File Name: COMP10-05.txt
Enter src & terminal: 0 9
Enter NB : 1024
N= 10 M = 90 type: U
READ OK
U NNNN=  10 
Grid 1024 1 1
DEVICE 0  sf = 4 sd = 8 
OKKKKKKKK 0x286100390 
TID SIZE = 15 MaskSize = 45 MAX_Q_SIZE = 100
BACK  rel[ 0] (Lower Bound) = 0.9957283895 
Eps[0] = 0.00028902866779390024 Upper Bound = 0.99601741811792976478 
NP[ 0] = 5317368 
Time = 6458.82 ms 
Totals: 0.99572838945013586454  0.00028902866779390024  5317368 



-------------- COMP25-05.txt ----------------

======== original ===============
TOTAL 47709159424  FREE 47427092480 
GPU Count : 1
HEAP 3221225472 
Enter Input File Name: COMP25-05.txt
Enter src & terminal: 0 24
Enter NB : 1024
N= 25 M = 600 type: U
READ OK
U NNNN=  25 
Grid 1024 1 1
DEVICE 0  sf = 4 sd = 8 
OKKKKKKKK 0x2c55c5e90 
TID SIZE = 15 MaskSize = 300 MAX_Q_SIZE = 100

BACK  rel[ 0] (Lower Bound) = 0.9995369421 
Eps[0] = 0.00046299825800404039 Upper Bound = 0.99999994039527784206 
NP[ 0] = 1090605865
Time = 175613 ms 
Totals: 0.99953694213727384277  0.00046299825800404039  1090605865 


============ new version ==================
(main) root@C.32228332:/workspace/relaibilityGPU/workingPad$ ./network_reliability ../seq/COMP25-05.txt 0 24 --queue_capacity 96777216
Warning: header E=600, found 300 unique edges
Graph loaded: N=25  E=300  mask_words=19  src=0  dst=24
Launch config: 168 blocks × 32 threads, smem=92 B, SMs=84

============================================
Two-Terminal Network Reliability Results
============================================
Graph:            N = 25, E = 300
Terminals:        src = 0, dst = 24
--------------------------------------------
Reliability:      R(G) = 0.168479358492
R + ESP:                 0.168479444447
ESP (truncated):         8.595436e-08
Truncation bound:        5.101774e-07
--------------------------------------------
Paths enumerated: 178673007
Nodes processed:  2437634047
Kernel iterations:51
Max queue RSS:    7383.516 MB  (96777216 items)
Total time:       213998.844 ms
============================================



----------- COMP15-05.txt--------------

========== Version 1 ============== 
=== Computation Complete ===
Total iterations: 569201259
Paths processed:  569201259
Time:             955.246 seconds
Rate:             595869 iter/sec

=== RESULTS ===
Reliability:       0.99971024480747128216
Truncation (Eps):  0.00019948614082994132
Total:             0.99990973094830126744
Paths enumerated:  569201259
Total time:        955404 ms
Max RSS:           2134932 KB


============= original ===================
TOTAL 47709159424  FREE 47427092480 
GPU Count : 1
HEAP 3221225472 
Enter Input File Name: COMP15-05.txt
Enter src & terminal: 0 14
Enter NB : 1024
N= 15 M = 210 type: U
READ OK
U NNNN=  15 
Grid 1024 1 1
DEVICE 0  sf = 4 sd = 8 
OKKKKKKKK 0x2c538cc90 
TID SIZE = 15 MaskSize = 105 MAX_Q_SIZE = 100
BACK  rel[ 0] (Lower Bound) = 0.9994680106 
Eps[0] = 0.00040983181633616212 Upper Bound = 0.99987784237113996877 
NP[ 0] = 90108256 
Time = 21463.3 ms 
Totals: 0.99946801055480383180  0.00040983181633616212  90108256 


========== New =======================
(main) root@C.32228332:/workspace/relaibilityGPU/workingPad$ ./network_reliability ../seq/COMP15-05.txt 0 14 --queue_capacity 96777216
Warning: header E=210, found 105 unique edges
Graph loaded: N=15  E=105  mask_words=7  src=0  dst=14
Launch config: 168 blocks × 32 threads, smem=92 B, SMs=84

============================================
Two-Terminal Network Reliability Results
============================================
Graph:            N = 15, E = 105
Terminals:        src = 0, dst = 14
--------------------------------------------
Reliability:      R(G) = 0.717891993596
R + ESP:                 0.717892079550
ESP (truncated):         8.595436e-08
Truncation bound:        1.197316e-07
--------------------------------------------
Paths enumerated: 147437316
Nodes processed:  2325563967
Kernel iterations:51
Max queue RSS:    7383.516 MB  (96777216 items)
Total time:       136037.203 ms
============================================



-------------- seq -----------------
=== Computation Complete ===
Total iterations: 723214750
Paths processed:  1196364013
Time:             3377.05 seconds
Rate:             214155 iter/sec

=== RESULTS ===
Reliability:       0.99974458824546252877
Truncation (Eps):  0.00016501023287821424
Total (rel + Eps): 0.99990959847834071095
Paths enumerated:  1196364013
Max RSS:           1538072 KB
Total time:        3.37729e+06 ms

=== Done ===
```
