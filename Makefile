all:main

CC=g++
CXXFLAGS=-g -std=c++17 -O3
INCLUDE=-I/usr/include/eigen3 -I./include
LIBRABY=-lssl -lcrypto -lgmp -lntl -lm -lgmpxx
PAILLIER=./paillier
PAILLIERTARGET=$(PAILLIER)/paillier.o $(PAILLIER)/paillierMatrix.o $(PAILLIER)/paillierGmpImp.o

$(PAILLIER)/%.o:$(PAILLIER)/%.cpp
	$(CC) $(INCLUDE) $(CXXFLAGS) -c $^ -o $@
main.o:main.cpp
	$(CC) $(INCLUDE) $(CXXFLAGS) -c $^ -o $@
main:main.o $(PAILLIERTARGET)
	$(CC) -o $@ $^ $(LIBRABY)
clean:
	find . -name '*.o' -type f -print -exec rm -rf {} \;
	rm main