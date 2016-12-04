FROM marcokuchla/llvm-clang

RUN apt-get update && apt-get install nano && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/local/src/llvm-build/llvm/lib/CodeGen/RAColorBasedCoalescing/
ADD CMakeLists.txt.CodeGen /usr/local/src/llvm-build/llvm/lib/CodeGen/CMakeLists.txt
RUN ln -s /usr/local/src/llvm-build/llvm/lib/CodeGen/RAColorBasedCoalescing/RAColorBasedCoalescing/RAColorBasedCoalescing.cpp /usr/local/src/llvm-build/llvm/lib/CodeGen/ 
CMD /bin/bash
