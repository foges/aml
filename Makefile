# Set variables here
CUDA_DIR=/usr/local/cuda

EXTRA_FLAGS=-DAML_DEBUG -g

CUDA_ARCH=sm_52

# OS Specific
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S), Linux)
  CC=g++
  CUDA_LIB_DIR=${CUDA_DIR}/lib64
  LDFLAGS=-lblas -pthread -fopenmp
  GTEST_DIR=/home/fougner/libs/googletest/googletest
endif
ifeq ($(UNAME_S), Darwin)
  CC=clang++
  CUDA_LIB_DIR=${CUDA_DIR}/lib
  LDFLAGS=-framework Accelerate
  GTEST_DIR=/Users/chris/code/libs/googletest/googletest
endif

# Compilation
FLAGS=-std=c++11 -Wall -Wextra -Wno-unknown-pragmas ${EXTRA_FLAGS}
DEPS=-Iinclude -isystem ${GTEST_DIR}/include

NVCC=nvcc -ccbin=${CC}
NVFLAGS=-std=c++11 -arch ${CUDA_ARCH} \
    --compiler-options -Wall,-Wextra,-Wno-unknown-pragmas \
    ${EXTRA_FLAGS} -DAML_GPU -x cu
NVDEPS=-isystem ${CUDA_DIR}/include
CUDA_LDFLAGS=-L${CUDA_LIB_DIR} -lcuda -lcudart -lcublas -lcusolver

TESTS=$(wildcard test/cpu/gtest_*.cpp)
NVTESTS=$(wildcard test/gpu/gtest_*.cpp)
OBJECTS=$(addprefix build/cpu/,$(notdir $(TESTS:.cpp=.o)))
NVOBJECTS=$(addprefix build/gpu/cpu/,$(notdir $(TESTS:.cpp=.o))) \
    $(addprefix build/gpu/gpu/,$(notdir $(NVTESTS:.cpp=.o)))

.PHONY: test
test: testcpu testgpu

.PHONY: testcpu
testcpu: build/cpu/test
	$^

.PHONY: testgpu
testgpu: build/gpu/test
	DYLD_LIBRARY_PATH=${CUDA_LIB_DIR} $^

.PHONY: build/cpu/test
build/cpu/test: ${OBJECTS} build/gtest/main.o build/gtest/libgtest.a
	${CC} $^ ${LDFLAGS} -o $@

.PHONY: build/gpu/test
build/gpu/test: ${NVOBJECTS} build/gtest/main.o build/gtest/libgtest.a
	${CC} $^ ${LDFLAGS} ${CUDA_LDFLAGS} -o $@

build/cpu/%.o: test/cpu/%.cpp build/cpu
	${CC} ${DEPS} ${FLAGS} $< -c -o $@

build/gpu/cpu/%.o: test/cpu/%.cpp build/gpu
	${NVCC} ${DEPS} ${NVDEPS} ${NVFLAGS} $< -c -o $@

build/gpu/gpu/%.o: test/gpu/%.cpp build/gpu
	${NVCC} ${DEPS} ${NVDEPS} ${NVFLAGS} $< -c -o $@

build/gtest/main.o: ${GTEST_DIR}/src/gtest_main.cc build/gtest
	${CC} -I${GTEST_DIR}/include $< -c -o $@

build/gtest/libgtest.a: build/gtest/gtest-all.o
	ar -r $@ $<

build/gtest/gtest-all.o: ${GTEST_DIR}/src/gtest-all.cc build/gtest
	${CC} -isystem ${GTEST_DIR}/include -I${GTEST_DIR} -pthread -c $< -o $@

build/gtest:
	mkdir -p $@

build/gpu:
	mkdir -p $@ $@/cpu $@/gpu

build/cpu:
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf build/cpu/ build/gpu/

