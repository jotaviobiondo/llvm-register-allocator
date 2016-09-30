mkdir test
cp hello.c test/hello.c
cd test
echo 'Compilando hello.c: "clang hello.c -o hello"'
clang hello.c -o hello
echo 'Compilando hello.c para llvm (object): "clang -O3 -emit-llvm hello.c -c -o hello.bc"'
clang -O3 -emit-llvm hello.c -c -o hello.bc
echo 'Compilando hello.c para llvm (assembly): "clang -O3 -emit-llvm hello.c -S -o hello.ll"'
clang -O3 -emit-llvm hello.c -S -o hello.ll
echo 'Executando hello'
./hello
echo 'Executando hello.bc'
lli hello.bc
echo 'Executando hello.ll'
lli hello.ll
echo 'Disassembling codigo: "llvm-dis < hello.bc"'
llvm-dis < hello.bc
echo 'Compilando o programa para codigo assembly nativo usando llc: "llc hello.bc -o hello.s"'
llc hello.bc -o hello.s
echo 'Usando gcc para compilar o codigo assembly em um programa: "gcc hello.s -o hello.native"'
gcc hello.s -o hello.native
echo 'Executando o codigo gerado pelo gcc'
./hello.native
cd ../
rm -rf test
