CC=clang++
GTEST_DIR=/Users/chris/code/libs/googletest/googletest
FLAGS=-std=c++11 -pthread -framework Accelerate -Wall -DAML_DEBUG #-DALL_GPU
DEPS=-Iinclude -I${GTEST_DIR}/include

TESTS=$(wildcard test/gtest_*.cpp)

.PHONY: test
test: build/cpu/test
	$^

.PHONY: build/cpu/test
build/cpu/test: ${TESTS} build/gtest/libgtest.a
	mkdir -p build/ build/cpu/
	${CC} ${DEPS} ${FLAGS} ${TESTS} ${GTEST_DIR}/src/gtest_main.cc build/gtest/libgtest.a -o $@

build/gtest/libgtest.a:
	mkdir -p build/ build/gtest/
	${CC} -isystem ${GTEST_DIR}/include -I${GTEST_DIR} -pthread -c ${GTEST_DIR}/src/gtest-all.cc -o build/gtest/gtest-all.o
	ar -rv $@ build/gtest/gtest-all.o

.PHONY: clean
clean:
	rm -rf build/cpu/ build/gpu/
