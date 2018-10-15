### Program configuration
BUFFER_SIZE="BUFFER_SIZE=10000"

### Change the following parameters if you are using a different library
PYTHON_CONFIG=`python3.6-config --includes --libs`
BOOST_PYTHONLIB=boost_python36

### Compile settings
CC=clang++
LANGFLAGS=-std=c++14
COMPFLAGS=-shared -fPIC -Wextra -O3
MACROFLAGS=-D$(BUFFER_SIZE) -DDEBUG
EXTLIB=$(PYTHON_CONFIG) -l$(BOOST_PYTHONLIB)
CCFLAGS=$(LANGFLAGS) $(COMPFLAGS) $(EXTLIB) $(MACROFLAGS)

cforward.so : cforward.cc array_wrapper.h
	$(CC) -o cforward.so cforward.cc $(CCFLAGS)

clean:
	rm *.so