.PHONY: clean all

CFLAGS = -std=gnu11 -O3 -g -fno-omit-frame-pointer -Wall -march=native -mavx512f -mcldemote

c2c-bench: c2c-bench.c
	cc $(CFLAGS) $< -o $@
