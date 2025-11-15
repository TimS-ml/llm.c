# ==============================================================================
# llm.c Makefile
# ==============================================================================
# This Makefile builds the llm.c training and testing executables for GPT-2
# models. It supports both CPU-only builds and CUDA GPU-accelerated builds.
#
# Main targets:
#   all              - Build all available targets (CPU and GPU if nvcc found)
#   train_gpt2       - CPU training executable
#   test_gpt2        - CPU testing executable
#   train_gpt2cu     - GPU training executable (requires CUDA)
#   test_gpt2cu      - GPU testing executable (requires CUDA)
#   clean            - Remove all build artifacts
#
# Environment variables:
#   CC               - C compiler (default: clang)
#   PRECISION        - Model precision: FP32, FP16, or BF16 (default: BF16)
#   USE_CUDNN        - Enable cuDNN flash attention (0 or 1, default: 0)
#   NO_OMP           - Disable OpenMP (0 or 1, default: 0)
#   NO_MULTI_GPU     - Disable multi-GPU NCCL support (0 or 1, default: 0)
#   NO_USE_MPI       - Disable MPI support (0 or 1, default: 0)
#
# Usage examples:
#   make                          # Build all targets with default settings
#   make train_gpt2cu USE_CUDNN=1 # Build GPU trainer with cuDNN
#   make PRECISION=FP32           # Build with FP32 precision
#   make clean                    # Clean build artifacts
# ==============================================================================

# C compiler configuration (can be overridden via environment)
CC ?= clang

# CPU compiler flags
# -Ofast: Maximum optimization including fast math
# -Wno-unused-result: Suppress warnings for unused return values
# -Wno-ignored-pragmas: Suppress warnings for unrecognized pragmas
# -Wno-unknown-attributes: Suppress warnings for unknown attributes
CFLAGS = -Ofast -Wno-unused-result -Wno-ignored-pragmas -Wno-unknown-attributes

# Linker flags and libraries
LDFLAGS =
LDLIBS = -lm  # Link against math library
INCLUDES =

# Conditional compiler flags (tested for support before being added)
# -march=native: Optimize for the current CPU architecture
CFLAGS_COND = -march=native

# ==============================================================================
# Platform Detection and Basic Configuration
# ==============================================================================

# Detect operating system
SHELL_UNAME = $(shell uname)

# Platform-specific commands for file operations
REMOVE_FILES = rm -f
OUTPUT_FILE = -o $@
CUDA_OUTPUT_FILE = -o $@

# ==============================================================================
# CUDA/NVCC Configuration
# ==============================================================================

# NVCC optimization level (0-3, where 0 = fastest compile, 3 = most optimized)
# Can be overridden: make FORCE_NVCC_O=0 for faster development iteration
FORCE_NVCC_O ?= 3

# NVCC (NVIDIA CUDA Compiler) flags
# --threads=0: Use all available CPU cores for compilation
# -t=0: Short form of --threads
# --use_fast_math: Enable fast math operations (may sacrifice precision for speed)
# -std=c++17: Use C++17 standard
# -O$(FORCE_NVCC_O): Optimization level
NVCC_FLAGS = --threads=0 -t=0 --use_fast_math -std=c++17 -O$(FORCE_NVCC_O)

# CUDA linker flags - link against cuBLAS libraries for matrix operations
NVCC_LDFLAGS = -lcublas -lcublasLt
NVCC_INCLUDES =
NVCC_LDLIBS =
NCLL_INCUDES =
NVCC_CUDNN =

# cuDNN support (NVIDIA's Deep Learning primitives library)
# By default disabled because it significantly increases compile time (seconds -> minute)
# Enable with: make USE_CUDNN=1
# Provides flash-attention implementation for better performance
USE_CUDNN ?= 0

# ==============================================================================
# Build Directory Configuration
# ==============================================================================

# Create build directory for object files
# This keeps the project root clean and organizes build artifacts
BUILD_DIR = build

# Platform-specific build directory creation
ifeq ($(OS), Windows_NT)
  $(shell if not exist $(BUILD_DIR) mkdir $(BUILD_DIR))
  REMOVE_BUILD_OBJECT_FILES := del $(BUILD_DIR)\*.obj
else
  $(shell mkdir -p $(BUILD_DIR))
  REMOVE_BUILD_OBJECT_FILES := rm -f $(BUILD_DIR)/*.o
endif

# ==============================================================================
# Utility Functions
# ==============================================================================

# Function to check if a program exists in the system PATH
# Used to detect available tools like nvcc, nvidia-smi, etc.
ifneq ($(OS), Windows_NT)
define file_exists_in_path
  $(which $(1) 2>/dev/null)
endef
else
define file_exists_in_path
  $(shell where $(1) 2>nul)
endef
endif

# ==============================================================================
# GPU Compute Capability Detection
# ==============================================================================

# Automatically detect GPU compute capability using nvidia-smi
# This ensures we compile for the correct GPU architecture
# Skip in CI environments where GPUs may not be available
ifneq ($(CI),true)
  ifndef GPU_COMPUTE_CAPABILITY
    ifneq ($(call file_exists_in_path, nvidia-smi),)
      # Query all GPUs for compute capability
      # Remove decimal points (e.g., 7.5 -> 75)
      # Sort numerically and select lowest (ensures compatibility with all GPUs)
      GPU_COMPUTE_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | sort -n | head -n 1)
      GPU_COMPUTE_CAPABILITY := $(strip $(GPU_COMPUTE_CAPABILITY))
    endif
  endif
endif

# Add GPU architecture flags if compute capability was detected or specified
# Format: compute_XX for PTX and sm_XX for binary code
# Can be manually overridden: make GPU_COMPUTE_CAPABILITY=75
ifneq ($(GPU_COMPUTE_CAPABILITY),)
  NVCC_FLAGS += --generate-code arch=compute_$(GPU_COMPUTE_CAPABILITY),code=[compute_$(GPU_COMPUTE_CAPABILITY),sm_$(GPU_COMPUTE_CAPABILITY)]
endif

# ==============================================================================
# Platform-Specific Configuration and Feature Detection
# ==============================================================================

$(info ---------------------------------------------)

ifneq ($(OS), Windows_NT)
  # Locate NVIDIA CUDA compiler on Unix-like systems
  NVCC := $(shell which nvcc 2>/dev/null)
  # Link against NVIDIA Management Library for GPU monitoring
  NVCC_LDFLAGS += -lnvidia-ml

  # Function to test if the C compiler supports a given flag
  # This allows us to conditionally add flags based on compiler capabilities
  define check_and_add_flag
    $(eval FLAG_SUPPORTED := $(shell printf "int main() { return 0; }\n" | $(CC) $(1) -x c - -o /dev/null 2>/dev/null && echo 'yes'))
    ifeq ($(FLAG_SUPPORTED),yes)
        CFLAGS += $(1)
    endif
  endef

  # Test and add conditional flags (like -march=native) if supported
  $(foreach flag,$(CFLAGS_COND),$(eval $(call check_and_add_flag,$(flag))))
else
  # Windows-specific configuration using MSVC compiler
  CFLAGS :=
  REMOVE_FILES = del *.exe,*.obj,*.lib,*.exp,*.pdb && del
  SHELL_UNAME := Windows

  # Detect CUDA compiler on Windows
  ifneq ($(shell where nvcc 2> nul),"")
    NVCC := nvcc
  else
    NVCC :=
  endif

  # Use Microsoft Visual C++ compiler
  CC := cl

  # MSVC compiler flags
  # /Idev: Include dev directory
  # /Zi: Generate debug information
  # /nologo: Suppress startup banner
  # /W4: Warning level 4
  # /O2 /Oi /Ot /GL: Optimization flags (maximize speed)
  # /fp:fast: Fast floating point model
  # /openmp:llvm: Enable OpenMP with LLVM runtime
  CFLAGS = /Idev /Zi /nologo /W4 /WX- /diagnostics:column /sdl /O2 /Oi /Ot /GL /D _DEBUG /D _CONSOLE /D _UNICODE /D UNICODE /Gm- /EHsc /MD /GS /Gy /fp:fast /Zc:wchar_t /Zc:forScope /Zc:inline /permissive- \
   /external:W3 /Gd /TP /wd4996 /Fd$@.pdb /FC /openmp:llvm
  LDFLAGS :=
  LDLIBS :=
  INCLUDES :=
  NVCC_FLAGS += -I"dev"

  # Different output file handling for CI vs local builds
  ifeq ($(WIN_CI_BUILD),1)
    $(info Windows CI build)
    OUTPUT_FILE = /link /OUT:$@
    CUDA_OUTPUT_FILE = -o $@
  else
    $(info Windows local build)
    OUTPUT_FILE = /link /OUT:$@ && copy /Y $@ $@.exe
    CUDA_OUTPUT_FILE = -o $@ && copy /Y $@.exe $@
  endif
endif

# ==============================================================================
# cuDNN Configuration (Flash Attention Support)
# ==============================================================================

# cuDNN provides optimized flash-attention implementation
# To use cuDNN:
#   1. Install cuDNN (see README for instructions)
#   2. Clone cudnn-frontend: git clone https://github.com/NVIDIA/cudnn-frontend.git
#   3. Build with: make train_gpt2cu USE_CUDNN=1
#
# You can override the cudnn-frontend path:
#   make USE_CUDNN=1 CUDNN_FRONTEND_PATH=/custom/path
#
# Default search paths:
#   - $HOME/cudnn-frontend/include
#   - ./cudnn-frontend/include
ifeq ($(USE_CUDNN), 1)
  ifeq ($(SHELL_UNAME), Linux)
    ifeq ($(shell [ -d $(HOME)/cudnn-frontend/include ] && echo "exists"), exists)
      $(info ✓ cuDNN found, will run with flash-attention)
      CUDNN_FRONTEND_PATH ?= $(HOME)/cudnn-frontend/include
    else ifeq ($(shell [ -d cudnn-frontend/include ] && echo "exists"), exists)
      $(info ✓ cuDNN found, will run with flash-attention)
      CUDNN_FRONTEND_PATH ?= cudnn-frontend/include
    else
      $(error ✗ cuDNN not found. See the README for install instructions and the Makefile for hard-coded paths)
    endif
    NVCC_INCLUDES += -I$(CUDNN_FRONTEND_PATH)
    NVCC_LDFLAGS += -lcudnn
    NVCC_FLAGS += -DENABLE_CUDNN
    NVCC_CUDNN = $(BUILD_DIR)/cudnn_att.o
  else
    ifneq ($(OS), Windows_NT)
      $(info → cuDNN is not supported on MAC OS right now)
    else
      $(info ✓ Windows cuDNN found, will run with flash-attention)
      ifeq ($(shell if exist "$(HOMEDRIVE)$(HOMEPATH)\cudnn-frontend\include" (echo exists)),exists)
        CUDNN_FRONTEND_PATH ?= $(HOMEDRIVE)$(HOMEPATH)\cudnn-frontend\include #override on command line if different location
      else ifeq ($(shell if exist "cudnn-frontend\include" (echo exists)),exists)
        CUDNN_FRONTEND_PATH ?= cudnn-frontend\include #override on command line if different location
      else
        $(error ✗ cuDNN not found. See the README for install instructions and the Makefile for hard-coded paths)
      endif
      CUDNN_INCLUDE_PATH ?= -I"C:\Program Files\NVIDIA\CUDNN\v9.1\include\12.4"
      CUDNN_FRONTEND_PATH += $(CUDNN_INCLUDE_PATH)
      NVCC_FLAGS += --std c++20 -Xcompiler "/std:c++20" -Xcompiler "/EHsc /W0 /nologo /Ox /FS" -maxrregcount=0 --machine 64
      NVCC_CUDNN = $(BUILD_DIR)\cudnn_att.obj
      NVCC_INCLUDES += -I$(CUDNN_FRONTEND_PATH)
      NVCC_LDFLAGS += -L"C:\Program Files\NVIDIA\CUDNN\v9.1\lib\12.4\x64" -lcudnn
      NVCC_FLAGS += -DENABLE_CUDNN
    endif
  endif
else
  $(info → cuDNN is manually disabled by default, run make with `USE_CUDNN=1` to try to enable)
endif

# ==============================================================================
# OpenMP Configuration (CPU Parallelization)
# ==============================================================================

# OpenMP enables multi-threaded CPU execution for significant performance gains
#
# Installation:
#   macOS:  brew install libomp
#   Ubuntu: sudo apt-get install libomp-dev
#
# Usage at runtime:
#   OMP_NUM_THREADS=8 ./train_gpt2
#
# To disable OpenMP:
#   make NO_OMP=1
#
# This section auto-detects OpenMP support and configures compiler flags
ifeq ($(NO_OMP), 1)
  $(info OpenMP is manually disabled)
else
  ifneq ($(OS), Windows_NT)
  # Detect if running on macOS or Linux
    ifeq ($(SHELL_UNAME), Darwin)
      # Check for Homebrew's libomp installation in different common directories
      ifeq ($(shell [ -d /opt/homebrew/opt/libomp/lib ] && echo "exists"), exists)
        # macOS with Homebrew on ARM (Apple Silicon)
        CFLAGS += -Xclang -fopenmp -DOMP
        LDFLAGS += -L/opt/homebrew/opt/libomp/lib
        LDLIBS += -lomp
        INCLUDES += -I/opt/homebrew/opt/libomp/include
        $(info ✓ OpenMP found)
      else ifeq ($(shell [ -d /usr/local/opt/libomp/lib ] && echo "exists"), exists)
        # macOS with Homebrew on Intel
        CFLAGS += -Xclang -fopenmp -DOMP
        LDFLAGS += -L/usr/local/opt/libomp/lib
        LDLIBS += -lomp
        INCLUDES += -I/usr/local/opt/libomp/include
        $(info ✓ OpenMP found)
      else
        $(info ✗ OpenMP not found)
      endif
    else
      # Check for OpenMP support in GCC or Clang on Linux
      ifeq ($(shell echo | $(CC) -fopenmp -x c -E - > /dev/null 2>&1; echo $$?), 0)
        CFLAGS += -fopenmp -DOMP
        LDLIBS += -lgomp
        $(info ✓ OpenMP found)
      else
        $(info ✗ OpenMP not found)
      endif
    endif
  endif
endif

# ==============================================================================
# NCCL Configuration (Multi-GPU Support)
# ==============================================================================

# NCCL (NVIDIA Collective Communications Library) enables multi-GPU training
# across multiple GPUs on the same node or across multiple nodes
#
# Installation on Ubuntu:
#   sudo apt install libnccl2 libnccl-dev
#
# To disable multi-GPU support:
#   make NO_MULTI_GPU=1
#
# Note: Not supported on macOS
ifeq ($(NO_MULTI_GPU), 1)
  $(info → Multi-GPU (NCCL) is manually disabled)
else
  ifneq ($(OS), Windows_NT)
    # Detect if running on macOS or Linux
    ifeq ($(SHELL_UNAME), Darwin)
      $(info ✗ Multi-GPU on CUDA on Darwin is not supported, skipping NCCL support)
    else ifeq ($(shell dpkg -l | grep -q nccl && echo "exists"), exists)
      $(info ✓ NCCL found, OK to train with multiple GPUs)
      NVCC_FLAGS += -DMULTI_GPU
      NVCC_LDLIBS += -lnccl
    else
      $(info ✗ NCCL is not found, disabling multi-GPU support)
      $(info ---> On Linux you can try install NCCL with `sudo apt install libnccl2 libnccl-dev`)
    endif
  endif
endif

# ==============================================================================
# MPI Configuration (Multi-Node Distributed Training)
# ==============================================================================

# MPI (Message Passing Interface) enables distributed training across multiple nodes
# Works in conjunction with NCCL for multi-GPU, multi-node setups
#
# Default OpenMPI installation path (can be overridden)
OPENMPI_DIR ?= /usr/lib/x86_64-linux-gnu/openmpi
OPENMPI_LIB_PATH = $(OPENMPI_DIR)/lib/
OPENMPI_INCLUDE_PATH = $(OPENMPI_DIR)/include/

# To disable MPI:
#   make NO_USE_MPI=1
#
# To specify custom MPI path:
#   make OPENMPI_DIR=/custom/openmpi/path
ifeq ($(NO_USE_MPI), 1)
  $(info → MPI is manually disabled)
else ifeq ($(shell [ -d $(OPENMPI_LIB_PATH) ] && [ -d $(OPENMPI_INCLUDE_PATH) ] && echo "exists"), exists)
  $(info ✓ MPI enabled)
  NVCC_INCLUDES += -I$(OPENMPI_INCLUDE_PATH)
  NVCC_LDFLAGS += -L$(OPENMPI_LIB_PATH)
  NVCC_LDLIBS += -lmpi
  NVCC_FLAGS += -DUSE_MPI
else
  $(info ✗ MPI not found)
endif

# ==============================================================================
# Precision Configuration
# ==============================================================================

# Numerical precision for model weights and computations
# Options:
#   BF16 (default) - BFloat16: Good balance of range and performance, recommended
#   FP16           - Float16: Faster but smaller range, may need careful tuning
#   FP32           - Float32: Full precision, slower but most stable
#
# Usage:
#   make PRECISION=FP32
#   make train_gpt2cu PRECISION=FP16
PRECISION ?= BF16
VALID_PRECISIONS := FP32 FP16 BF16

# Validate precision setting
ifeq ($(filter $(PRECISION),$(VALID_PRECISIONS)),)
  $(error Invalid precision $(PRECISION), valid precisions are $(VALID_PRECISIONS))
endif

# Set precision flags
ifeq ($(PRECISION), FP32)
  PFLAGS = -DENABLE_FP32
else ifeq ($(PRECISION), FP16)
  PFLAGS = -DENABLE_FP16
else
  PFLAGS = -DENABLE_BF16
endif

# ==============================================================================
# Build Targets
# ==============================================================================

# PHONY targets are always executed (not treated as files)
.PHONY: all train_gpt2 test_gpt2 train_gpt2cu test_gpt2cu train_gpt2fp32cu test_gpt2fp32cu profile_gpt2cu

# Start with CPU-only targets
TARGETS = train_gpt2 test_gpt2

# Add GPU targets if CUDA compiler is available
ifeq ($(NVCC),)
    $(info ✗ nvcc not found, skipping GPU/CUDA builds)
else
    $(info ✓ nvcc found, including GPU/CUDA support)
    TARGETS += train_gpt2cu test_gpt2cu train_gpt2fp32cu test_gpt2fp32cu $(NVCC_CUDNN)
endif

$(info ---------------------------------------------)

# Default target: build all available executables
all: $(TARGETS)

# ==============================================================================
# CPU Build Targets
# ==============================================================================

# Build CPU-only training executable
# Uses OpenMP for parallelization if available
# Usage: ./train_gpt2 -i <input_data>
train_gpt2: train_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

# Build CPU-only testing executable
# For running model tests and validations
# Usage: ./test_gpt2
test_gpt2: test_gpt2.c
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ $(LDLIBS) $(OUTPUT_FILE)

# ==============================================================================
# GPU Build Targets
# ==============================================================================

# Build cuDNN attention object file (if cuDNN is enabled)
# This provides optimized flash-attention implementation
$(NVCC_CUDNN): llmc/cudnn_att.cpp
	$(NVCC) -c $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_INCLUDES) -o $@

# Build GPU training executable with configurable precision
# Supports multi-GPU via NCCL and multi-node via MPI
# Usage: ./train_gpt2cu -i <input_data>
# Multi-GPU: mpirun -np 8 ./train_gpt2cu -i <input_data>
train_gpt2cu: train_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

# Build GPU training executable with FP32 precision only
# Legacy target for full-precision training
# Usage: ./train_gpt2fp32cu -i <input_data>
train_gpt2fp32cu: train_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

# Build GPU testing executable with configurable precision
# For running model tests and validations on GPU
# Usage: ./test_gpt2cu
test_gpt2cu: test_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

# Build GPU testing executable with FP32 precision only
# Legacy target for full-precision testing
# Usage: ./test_gpt2fp32cu
test_gpt2fp32cu: test_gpt2_fp32.cu
	$(NVCC) $(NVCC_FLAGS) $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS) $(CUDA_OUTPUT_FILE)

# Build GPU profiling executable
# Includes line information for NVIDIA profiling tools (nsys, nvprof)
# Usage: nsys profile ./profile_gpt2cu
profile_gpt2cu: profile_gpt2.cu $(NVCC_CUDNN)
	$(NVCC) $(NVCC_FLAGS) $(PFLAGS) -lineinfo $^ $(NVCC_LDFLAGS) $(NVCC_INCLUDES) $(NVCC_LDLIBS)  $(CUDA_OUTPUT_FILE)

# ==============================================================================
# Cleanup Target
# ==============================================================================

# Remove all build artifacts
# Usage: make clean
clean:
	$(REMOVE_FILES) $(TARGETS)
	$(REMOVE_BUILD_OBJECT_FILES)
