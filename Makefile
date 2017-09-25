cpu:
	mkdir -p build/ build/cpu/
	g++ test/eval_cpu.cpp -Iinclude -std=c++14 -o build/cpu/eval

gpu:
	mkdir -p build/ build/gpu/
	g++ test/eval_gpu.cpp -Iinclude -DAML_GPU -std=c++14 -o build/gpu/eval

clean:
	rm -rf build/
