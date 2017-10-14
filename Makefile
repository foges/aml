# Set variables here
CUDA_DIR=/usr/local/cuda
GTEST_DIR=/Users/chris/code/libs/googletest/googletest

EXTRA_FLAGS=-DAML_DEBUG

CUDA_ARCH=sm_53

# Compilation
CC=clang++
FLAGS=-std=c++11 -Wall -Wextra ${EXTRA_FLAGS}
LDFLAGS=-framework Accelerate
DEPS=-Iinclude -isystem ${GTEST_DIR}/include

NVCC=nvcc
NVFLAGS=-std=c++11 -arch ${CUDA_ARCH} --compiler-options -Wall,-Wextra \
    ${EXTRA_FLAGS} -DAML_GPU -x cu
NVDEPS=-isystem ${CUDA_DIR}/include
CUDA_FLAGS=-L${CUDA_DIR}/lib -lcuda -lcudart -lcublas

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
	DYLD_LIBRARY_PATH=${CUDA_DIR}/lib $^

.PHONY: build/cpu/test
build/cpu/test: ${OBJECTS} build/gtest/libgtest.a build/gtest/main.o
	${CC} ${LDFLAGS} $^ -o $@

.PHONY: build/gpu/test
build/gpu/test: ${NVOBJECTS} build/gtest/libgtest.a build/gtest/main.o
	${CC} ${LDFLAGS} ${CUDA_FLAGS} $^ -o $@

build/cpu/%.o: test/cpu/%.cpp build/cpu
	${CC} ${DEPS} ${FLAGS} $< -c -o $@

build/gpu/cpu/%.o: test/cpu/%.cpp build/gpu
	${NVCC} ${DEPS} ${NVDEPS} ${NVFLAGS} $< -c -o $@

build/gpu/gpu/%.o: test/gpu/%.cpp build/gpu
	${NVCC} ${DEPS} ${NVDEPS} ${NVFLAGS} $< -c -o $@

build/gtest/main.o: ${GTEST_DIR}/src/gtest_main.cc
	${CC} -I${GTEST_DIR}/include $^ -c -o $@

build/gtest/libgtest.a: build/gtest
	${CC} -isystem ${GTEST_DIR}/include -I${GTEST_DIR} -pthread -c \
	    ${GTEST_DIR}/src/gtest-all.cc -o build/gtest/gtest-all.o
	ar -rv $@ build/gtest/gtest-all.o

build/gtest:
	mkdir -p $@

build/gpu:
	mkdir -p $@ $@/cpu $@/gpu

build/cpu:
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf build/cpu/ build/gpu/

