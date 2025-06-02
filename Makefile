TARGET = ./bin/pcg
CC = gcc 
CFLAGS += -std=c99 -O3 -I./src -I./include -I./libs/fft/include -I./libs/gf64/include -I./libs/gf128/include -I./libs/base-ary-dpf/include -I/usr/include/openssl/
LDFLAGS = -march=native -lcrypto -lssl -lm -maes -ffast-math

# Define source files and filter out library files 
FFT_SRC = $(filter-out ./libs/fft/src/test.c, $(wildcard ./libs/fft/src/*.c))
DPF_SRC = $(filter-out ./libs/base-ary-dpf/src/test.c, $(wildcard ./libs/base-ary-dpf/src/*.c))
GF128_SRC = $(filter-out ./libs/gf128/src/test.c, $(wildcard ./libs/gf128/src/*.c))
GF64_SRC = $(filter-out ./libs/gf64/src/test.c, $(wildcard ./libs/gf64/src/*.c))
PCG_SRC = $(wildcard ./src/*.c)

all: $(TARGET)

OBJECTS = $(FFT_SRC:.c=.o) $(DPF_SRC:.c=.o) $(GF128_SRC:.c=.o) $(GF64_SRC:.c=.o) $(PCG_SRC:.c=.o)

$(TARGET): $(OBJECTS)
	@mkdir -p ./bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f libs/fft/src/*.o libs/base-ary-dpf/src/*.o *.o $(TARGET) $(OBJECTS)

.PHONY: all clean