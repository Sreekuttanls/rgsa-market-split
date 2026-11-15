CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O3 -march=native -ffp-contract=off -Wall -Wextra -Wpedantic

all: rgsa fj_driver

rgsa: rgsa.cpp qoblib.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

fj_driver: fj_driver.cpp qoblib.hpp ../third_party/feasibilityjump/feasibilityjump.hh
	$(CXX) $(CXXFLAGS) -Wno-unused-parameter -Wno-sign-compare -I../third_party/feasibilityjump -o $@ $<

clean:
	rm -f rgsa fj_driver

.PHONY: all clean
