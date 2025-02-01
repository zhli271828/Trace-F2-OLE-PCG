TARGET = ./bin/gf128
CC = gcc
CFLAGS += -std=c99 -O3 -I./include -I/usr/include/openssl/
LDFLAGS = -march=native -lcrypto -lssl -lm -maes -ffast-math

SRC = $(wildcard ./src/*.c)
OBJECTS = $(SRC:.c=.o)

$(TARGET): $(OBJECTS)
	@mkdir -p ./bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: all clean