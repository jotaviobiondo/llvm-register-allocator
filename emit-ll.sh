#!/bin/bash

filename="${1%.*}"
clang -S -emit-llvm $filename.c -o $filename.ll
