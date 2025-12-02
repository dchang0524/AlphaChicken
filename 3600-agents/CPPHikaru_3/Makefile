# Makefile for CPPHikaru_3
# Alternative to CMake - can be used directly with make

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra -fPIC
LDFLAGS = -shared

# Source files
SOURCES = cpphikaru3.cpp \
          agent_wrapper.cpp \
          search.cpp \
          evaluation.cpp \
          voronoi.cpp \
          bfs.cpp \
          hmm.cpp \
          zobrist.cpp \
          game_rules.cpp

OBJECTS = $(SOURCES:.cpp=.o)

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIBNAME = libcpphikaru3.so
endif
ifeq ($(UNAME_S),Darwin)
    LIBNAME = libcpphikaru3.dylib
    LDFLAGS += -install_name @rpath/libcpphikaru3.dylib
endif
ifeq ($(OS),Windows_NT)
    LIBNAME = cpphikaru3.dll
    CXX = g++
endif

.PHONY: all clean

all: $(LIBNAME)

$(LIBNAME): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS)
	@echo "Built successfully: $@"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(LIBNAME) libcpphikaru3.so libcpphikaru3.dylib cpphikaru3.dll

