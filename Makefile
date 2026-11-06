CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++20 -pthread

SRCS := $(wildcard src/*.cpp)
BINS := $(patsubst src/%.cpp,bin/%,$(SRCS))

all: $(BINS)

bin/%: src/%.cpp | bin
	$(CXX) $(CXXFLAGS) $< -o $@

bin:
	mkdir -p bin

clean:
	rm -rf bin

.PHONY: all clean
