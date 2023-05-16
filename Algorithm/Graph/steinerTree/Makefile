MATLABDIR=/usr/local/MATLAB/R2016b

CXXFLAGS=-Wall -O3 -g -Wno-deprecated -fopenmp -fPIC
#CXXFLAGS=-Wall -O3 -g -Wno-deprecated
LDFLAGS=-lboost_program_options -lgomp
DIST=*.?pp Makefile README LICENSE matlab
#CXXFLAGS=-Wall -O0 -g -ffloat-store
#CXXFLAGS=-Wall -O2 -g -ffloat-store
VER=1.3

BIN=msgsteiner ${MEXOBJ}



all: $(BIN)

tgz: clean
	mkdir msgsteiner-${VER} &&\
	cp -r ${DIST} msgsteiner-${VER}&&\
	tar -czf msgsteiner-${VER}.tgz msgsteiner-${VER}&&\
	rm -rf msgsteiner-${VER}

msgsteiner: msgsteiner.cpp ms.o mes.o proba.hpp
	g++ ${CXXFLAGS} msgsteiner.cpp ${LDFLAGS} ms.o mes.o -o msgsteiner
clean:
	rm -f $(BIN) *.o matlab/*.mexa64 matlab/*.mexglx .mlab *.tgz
mes.o: mes.cpp mes.hpp proba.hpp
	g++ ${CXXFLAGS} -c mes.cpp -o mes.o
ms.o: ms.cpp ms.hpp proba.hpp
	g++ ${CXXFLAGS} -c ms.cpp -o ms.o

.mlab: matlab.cpp ms.o mes.o
	${MATLABDIR}/bin/mex matlab.cpp -output matlab/MsgSteiner -v -g -O -largeArrayDims -lut "CXXFLAGS=\$$CXXFLAGS ${CXXFLAGS}" "LDFLAGS=\$$LDFLAGS ${LDFLAGS} ms.o mes.o -lgomp" \
	&& touch .mlab

.PHONY: matlab

matlab: .mlab
