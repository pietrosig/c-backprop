CC=gcc
CFLAGS=-Wall -Wextra -std=c11
SRC=$(wildcard src/*.c)
OBJ=$(filter-out src/test.c src/main.c, $(SRC))
TARGET=c_backprop
TEST_TARGET=test_bin

all: $(TARGET)

$(TARGET): $(OBJ) src/main.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

test: $(OBJ) src/test.c
	$(CC) $(CFLAGS) $^ -o $(TEST_TARGET) -lm

clean:
	rm -f $(TARGET) $(TEST_TARGET)
