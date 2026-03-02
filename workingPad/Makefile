# ============================================================
#  Makefile — CUDA Two-Terminal Network Reliability
# ============================================================

NVCC       ?= nvcc
ARCH       ?= sm_70          # change to your GPU arch (sm_75, sm_80, sm_86, sm_89, sm_90)
MASK_WORDS ?= 16             # supports up to MASK_WORDS*16 edges (16→256, 32→512, 256→4096)

NVCCFLAGS  = -O3 -arch=$(ARCH) \
             -DMAX_MASK_WORDS=$(MASK_WORDS) \
             --expt-relaxed-constexpr \
             -Xcompiler -Wall

TARGET     = network_reliability
SRC        = main.cu

# ---- build ----
all: $(TARGET)

$(TARGET): $(SRC) common.h graph.h device_code.cuh
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(TARGET)

# ---- convenience targets ----
run_simple: $(TARGET)
	./$(TARGET) test_simple.txt 0 3

run_bridge: $(TARGET)
	./$(TARGET) test_bridge.txt 0 4

run_kite: $(TARGET)
	./$(TARGET) test_kite.txt 0 3

# ---- clean ----
clean:
	rm -f $(TARGET)

.PHONY: all clean run_simple run_bridge run_kite
