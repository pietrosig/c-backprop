CC=gcc
CFLAGS=-Wall -Wextra -std=c11
SRC=$(wildcard src/*.c)
OBJ=$(filter-out src/test_numeric.c src/main.c, $(SRC))
TARGET=c_backprop
TEST_TARGET=test_numeric_bin

all: $(TARGET)

$(TARGET): $(OBJ) src/main.c
	$(CC) $(CFLAGS) $^ -o $@ -lm

test-numeric: $(OBJ) src/test_numeric.c
	$(CC) $(CFLAGS) $^ -o $(TEST_TARGET) -lm

clean:
	rm -f $(TARGET) $(TEST_TARGET)
