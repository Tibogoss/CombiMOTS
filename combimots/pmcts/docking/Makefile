# CombiMOTS QuickVina-GPU Makefile template.
# Override paths at invocation if needed, for example:
#   make source CONDA_ENV=$$CONDA_PREFIX OPENCL_LIB_PATH=/usr/local/cuda
WORK_DIR=$(CURDIR)
CONDA_ENV?=$(CONDA_PREFIX)
BOOST_LIB_PATH?=$(CONDA_ENV)
OPENCL_LIB_PATH?=$(CONDA_ENV)
CXX=g++
CXXFLAGS?=-O3

# Version and platform settings
OPENCL_VERSION=-DOPENCL_3_0
GPU_PLATFORM=-DNVIDIA_PLATFORM
DOCKING_BOX_SIZE=-DSMALL_BOX

# Include paths
BOOST_INC_PATH=-I$(BOOST_LIB_PATH)/include -I$(BOOST_LIB_PATH) -I$(BOOST_LIB_PATH)/boost
VINA_GPU_INC_PATH=-I$(WORK_DIR)/lib -I$(WORK_DIR)/OpenCL/inc
OPENCL_INC_PATH=-I$(OPENCL_LIB_PATH)/include

# Library paths and flags
LIB1=-lboost_program_options -lboost_system -lboost_filesystem -lboost_thread -lboost_serialization -lOpenCL
LIB2=-lstdc++ -lstdc++fs
LIB3=-lm -lpthread
LIB_PATH=-L$(BOOST_LIB_PATH)/lib -L$(BOOST_LIB_PATH)/stage/lib -L$(OPENCL_LIB_PATH)/lib -L$(OPENCL_LIB_PATH)/lib64

# Source files
SRC=./lib/*.cpp ./OpenCL/src/wrapcl.cpp

# Compilation flags
MACRO=$(OPENCL_VERSION) $(GPU_PLATFORM) $(DOCKING_BOX_SIZE) -DBOOST_TIMER_ENABLE_DEPRECATED

all:out
out:./main/main.cpp
	$(CXX) -o QuickVina2-GPU-2-1 $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) ./main/main.cpp $(CXXFLAGS) $(SRC) $(LIB_PATH) $(LIB1) $(LIB2) $(LIB3) $(MACRO) $(OPTION) -DNDEBUG

source:./main/main.cpp
	$(CXX) -o QuickVina2-GPU-2-1 $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) ./main/main.cpp $(CXXFLAGS) $(SRC) $(LIB_PATH) $(LIB1) $(LIB2) $(LIB3) $(MACRO) $(OPTION) -DNDEBUG -DBUILD_KERNEL_FROM_SOURCE

debug:./main/main.cpp
	$(CXX) -o QuickVina2-GPU-2-1 $(BOOST_INC_PATH) $(VINA_GPU_INC_PATH) $(OPENCL_INC_PATH) ./main/main.cpp -g $(SRC) $(LIB_PATH) $(LIB1) $(LIB2) $(LIB3) $(MACRO) $(OPTION) -DBUILD_KERNEL_FROM_SOURCE

clean:
	rm -f QuickVina2-GPU-2-1
