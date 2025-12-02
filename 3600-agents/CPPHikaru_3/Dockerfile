FROM ubuntu:22.04

# Install build tools
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source files
COPY *.cpp *.h Makefile ./

# Build the library
RUN make clean && make

# The library will be at /build/libcpphikaru3.so

