GTEST_DIR=/Users/chris/code/libs/googletest/googletest
FLAGS=-std=c++11 -pthread #-DALL_GPU
DEPS=-Iinclude -I${GTEST_DIR}/include

TESTS=$(wildcard test/gtest_*.cpp)

all: ${TESTS} build/gtest/libgtest.a
	mkdir -p build/ build/cpu/
	g++ ${DEPS} ${FLAGS} ${TESTS} ${GTEST_DIR}/src/gtest_main.cc build/gtest/libgtest.a -o build/cpu/eval

build/gtest/libgtest.a:
	mkdir -p build/ build/gtest/
	g++ -isystem ${GTEST_DIR}/include -I${GTEST_DIR} -pthread -c ${GTEST_DIR}/src/gtest-all.cc -o build/gtest/gtest-all.o
	ar -rv $@ build/gtest/gtest-all.o

clean:
	rm -rf build/cpu/ build/gpu/
