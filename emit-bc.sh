#!/bin/bash

filename="${1%.*}"
clang -c -emit-llvm $filename.c -o $filename.bc
