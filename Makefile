llvm:
	./compile.sh
compile:
	clang -c -emit-llvm tests/main.c -o tests/main.bc
	llc -regalloc=myregalloc tests/main.bc -o tests/main.s
run:
	./compile.sh
	clang -c -emit-llvm tests/main.c -o tests/main.bc
	llc -regalloc=myregalloc tests/main.bc -o tests/main.s
	
