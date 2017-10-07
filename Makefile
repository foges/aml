# Set variables here
CUDA_DIR=/usr/local/cuda/lib
GTEST_DIR=/Users/chris/code/libs/googletest/googletest

EXTRA_FLAGS=-DAML_DEBUG

CUDA_ARCH=sm_53

# Compilation
CC=clang++
FLAGS=-std=c++11 -Wall -Wextra ${EXTRA_FLAGS}
LDFLAGS=-framework Accelerate

NVCC=nvcc
NVFLAGS=-std=c++11 -arch ${CUDA_ARCH} --compiler-options -Wall,-Wextra \
    ${EXTRA_FLAGS} -DAML_GPU -x cu
CUDA_FLAGS=-L${CUDA_DIR} -lcuda -lcudart

DEPS=-Iinclude -I${GTEST_DIR}/include

TESTS=$(wildcard test/gtest_*.cpp)
OBJECTS=$(addprefix build/cpu/,$(notdir $(TESTS:.cpp=.o)))
NVOBJECTS=$(addprefix build/gpu/,$(notdir $(TESTS:.cpp=.o)))

.PHONY: test
test: testcpu testgpu

.PHONY: testcpu
testcpu: build/cpu/test
	$^

.PHONY: testgpu
testgpu: build/gpu/test
	DYLD_LIBRARY_PATH=${CUDA_DIR} $^

.PHONY: build/cpu/test
build/cpu/test: ${OBJECTS} build/gtest/libgtest.a build/gtest/main.o
	${CC} ${LDFLAGS} $^ -o $@

.PHONY: build/gpu/test
build/gpu/test: ${NVOBJECTS} build/gtest/libgtest.a build/gtest/main.o
	${CC} ${LDFLAGS} ${CUDA_FLAGS} $^ -o $@

build/cpu/%.o: test/%.cpp build/cpu
	${CC} ${DEPS} ${FLAGS} $< -c -o $@

build/gpu/%.o: test/%.cpp build/gpu
	${NVCC} ${DEPS} ${NVFLAGS} $< -c -o $@

build/gtest/main.o: ${GTEST_DIR}/src/gtest_main.cc
	${CC} -I${GTEST_DIR}/include $^ -c -o $@

build/gtest/libgtest.a: build/gtest
	${CC} -isystem ${GTEST_DIR}/include -I${GTEST_DIR} -pthread -c \
	    ${GTEST_DIR}/src/gtest-all.cc -o build/gtest/gtest-all.o
	ar -rv $@ build/gtest/gtest-all.o

build/gtest:
	mkdir -p $@

build/gpu:
	mkdir -p $@

build/cpu:
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf build/cpu/ build/gpu/

