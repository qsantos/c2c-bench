Core-to-core Throughput Benchmark
=================================

A simple benchmark program to evaluate core-to-core throughput on Intel
and AMD processors running Linux.

Building the benchmark
----------------------

Run `make` with either `clang` or `gcc` available.

Running the benchmark
---------------------

The benchmark must be run as follows:

```
c2c-bench mode hugepage-type \
          producer-type producer-cpu \
          consumer-type consumer-cpu \
          page-size working-set-size total-size
```

The arguments are:

* `mode`
  * `process`: execute the producer and consumer in separate processes.
  * `thread`: execute the producer and consumer in threads of one process.
* `hugepage-type`
  * `explicit`: specify a huge page size while allocating memory. This
    requires a sufficient number of huge pages to be reserved using
    `/sys/kernel/mm/hugepages/`. See
    [HugeTLB Pages](https://docs.kernel.org/admin-guide/mm/hugetlbpage.html)
    in the kernel documentation for details.
  * `transparent`: rely on the kernel to allocate transparent huge pages.
    The kernel may fall back to using 4 KiB pages if it cannot allocate
    enough huge pages, e.g. due to fragmentation.
* `producer-type`
  * `plain`: uses 512-bit stores to write data
  * `cldemote`: uses the `CLDEMOTE` instruction to force writes into the
     L3 cache. On microarchitectures without CLDEMOTE support, the CLDEMOTE
     instruction is treated as a no-op.
* `producer-cpu`: processor number on which to run the producer task.
* `consumer-type`
  * `plain`: uses 512-bit loads to read data
  * `cldemote`: uses the `CLDEMOTE` instruction to evict data from the
     core-private cache immediately after reading it. On microarchitectures
     without CLDEMOTE support, the CLDEMOTE instruction is treated as a no-op.
* `consumer-cpu`: processor number on which to run the consumer task.
* `page-size`: page size in bytes to use for memory allocated by the benchmark.
  If `hugepage-type` is `explicit`, then the page size must correspond to
  a hardware page size; otherwise, the page size is only used as an alignment
  hint.
  
  Sizes may be written with a `k`, `M`, or `G` suffix: `4k` is 4096 bytes,
  `2M` is 2097152 bytes, and `1G` is 1073741824 bytes.
* `working-set-size`: total amount of memory used by the benchmark.
  This is used to keep the data in a specific level of the cache hierarchy.
  Keep in mind that each iteration of the producer and consumer operates on
  a chunk of data equal to half of the working set -- see c2c-bench.c for
  details.
  
  As with the `page-size` option, sizes may be written with a 'k', `M`, or `G`
  suffix.
* `target-size`: target amount of data to transfer between the producer and
  consumer. This controls how long the benchmark runs before terminating.
  The actual amount of data transferred will be adjusted to an integer multiple
  of the amount of the segment size.

Using the included scripts
--------------------------

A few scripts are included for convenience:

* `./benchmark.sh producer-cpu consumer-cpu` runs the benchmark on the specified
  CPU numbers with a variety of working set sizes, and both 4k and 2M pages.
  The results are written to standard output as a CSV.

* `./plot.sh microarchitecture-name producer-cpu-name consumer-cpu-name` plots
  the data emitted by benchmark.sh using gnuplot. The plot is written to
  standard output as a PNG.
  
  The microarchitecture name, producer CPU name, and consumer CPU name are used
  to generate the plot title.

To benchmark transferring data between hyperthreads of a single Zen4c core,
you would execute the following:

```sh
./benchmark.sh 0 8 |
  ./plot.sh Zen4c 'Core 0, Thread 0' 'Core 0, Thread 1' >zen4c-core.png
```
