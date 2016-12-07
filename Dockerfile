FROM llvm_working

#RUN apt-get update && apt-get install nano vim --no-install-recommends -y && rm -rf /var/lib/apt/lists/*
WORKDIR /root
#ADD CMakeLists.txt.CodeGen llvm/lib/CodeGen/CMakeLists.txt
#RUN cd llvm/lib/CodeGen && ln -s /root/RAColorBasedCoalescing/RAColorBasedCoalescing.cpp
CMD /bin/bash
