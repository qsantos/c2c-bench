// SPDX-License-Identifier: MIT
//
// A simple core-to-core throughput benchmark. See README.md for usage.
//
// Internally, the benchmark is implemented as a producer and consumer that
// use a shared-memory queue. The queue is implemented as a pair of equally-
// sized segments, so that the producer can write one segment while the
// consumer is reading the other. The total size of the two segments, plus
// the space used for mutual exclusion, is equal to the user-configured
// working-set size.
//
// The producer and consumer work according to the pseudo-code below.
//
// producer:
// for (int i = 0; i < iterations; i++) {
//   int segment_no = i % 2;
//   acquire(segment_no, PRODUCER); // Wait for segment to be available
//   u64* s = segments[segment_no];
//   for (int j = 0; j < segment_size / 8; j++)
//     s[j] = i;
//   release(segment_no, CONSUMER); // Transfer segment to consumer
// }
//
// consumer:
// u64 sum[8] = {0, 0, 0, 0, 0, 0, 0, 0};
// for (int i = 0; i < iterations; i++) {
//   int segment_no = i % 2;
//   acquire(segment_no, CONSUMER); // Wait for segment
//   u64* s = segments[segment_no];
//   for (int j = 0; j < segment_size / 8; j++)
//     sum[j % 8] += s[j];
//   release(segment_no, PRODUCER); // Transfer to producer
// }
//
// The implementation uses AVX-512 and unrolling to so that the producer and
// consumer can execute as many stores and loads per cycle as the
// microarchitecture permits.

#define _GNU_SOURCE

#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

// Number of parallel accumulators used by the consumer.
//
// Using more than one accumulator avoids limiting the the consumer to a single
// 512-bit read per cycle, though this is not a concern on Zen 4 or Sapphire
// Rapids.
#define STRIPES 8

enum {
  // The buffer contains no data, and is ready to be written.
  OWNER_PRODUCER,

  // All entries in the buffer contain data, and are ready to be read.
  OWNER_CONSUMER
};

// Current owner of a data segment. Used for mutual exclusion without locking.
// To manipulate segment owners, see acquire() and release().
struct owner {
  // The current owner of the buffer; either OWNER_PRODUCER or OWNER_CONSUMER.
  atomic_uchar val;

  // Enough padding to fill the rest of the cache line and avoid false sharing
  // between adjacent owner objects.
  char padding[63];
};

// The working set for a benchmark run. This includes all memory that will be
// touched during the benchmark, including ownership information.
struct working_set {
  // The owner of each segment of the buffer.
  struct owner owner[2];

  // The data in the buffer. The data is logically divided into two segments
  // of 'len' bytes each, where 'len' is set in 'struct params'.
  // 'owner[0]' owns 'data[0..len-1]', and 'owner[1]' owns 'data[len..2*len-1]'.
  char data[];
};

// Benchmark results, as collected by the consumer task.
struct result {
  // A eight-wide summation of all 64-bit words received by the consumer.
  // This is used to check that the producer and consumer are working properly.
  __m512i sum;

  // The amount of time spent in the consumer loop.
  struct timespec elapsed;
};

// A region of memory shared with the producer and consumer tasks.
union shmem {
  // The buffer between the producer and consumer.
  struct working_set buffer;

  // The benchmark results from the consumer. Written after both the
  // producer and consumer have finished running, and the buffer will
  // not be used again.
  struct result result;
};

enum {
  // The consumer and producer run in different processes.
  MODE_PROCESS,

  // The consumer and producer run in different threads.
  MODE_THREAD,
};

enum {
  // Request the kernel to allocate huge pages explicitly.
  HUGEPAGE_EXPLICIT,

  // Rely on the kernel to use transparent huge pages in aligned regions.
  HUGEPAGE_TRANSPARENT
};

// An alias for a function pointer that can be used to spawn a new
// process or thread.
typedef void* (*task_t)(void*);

// Runtime parameters for a benchmark run.
struct params {
  // Shared memory region used by the benchmark.
  union shmem* mem;

  // Pointer to the implementations of the producer and consumer tasks.
  task_t producer;
  task_t consumer;

  // Number of segments transferred from producer to consumer.
  size_t iterations;

  // Length in bytes of the segments transferred from producer to consumer.
  size_t len;

  // Page size hint. This is used for alignment purposes; we rely on the kernel
  // to allocate transparent huge pages for aligned blocks of memory.
  size_t page_size;

  // MODE_PROCESS or MODE_THREAD
  unsigned mode;

  // HUGEPAGE_EXPLICIT or HUGEPAGE_TRANSPARENT
  unsigned hugepage_type;

  // CPUs to assign the producer and consumer tasks to.
  unsigned producer_cpu;
  unsigned consumer_cpu;
};

// Acquires exclusive access for 'owner' to the object guarded by 'o'.
// Access must be granted by the prior owner by calling 'release(o, owner)'.
__attribute__((always_inline))
static inline void acquire(struct owner* o, unsigned char owner) {
  while (atomic_load_explicit(&o->val, memory_order_acquire) != owner) {}
}

// Releases exclusive access of 'o' to 'new_owner'.
// The caller must have already acquired access with acquire().
__attribute__((always_inline))
static inline void release(struct owner* o, unsigned char new_owner) {
  atomic_store_explicit(&o->val, new_owner, memory_order_release);
}

// Pins the current thread to the specified CPU.
static void pin_to_cpu(unsigned cpu) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(cpu, &mask);
  int r = pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask);
  if (r) {
    fprintf(stderr, "pthread_setaffinity_np: %s", strerror(r));
    exit(1);
  }
}

// Returns the current time according to the system's monotonic clock.
//
// This time is suitable for measuring durations since does not jump or
// slew in response to time updates from NTP.
static struct timespec monotonic_time(void) {
  struct timespec result;
  if (clock_gettime(CLOCK_MONOTONIC, &result)) {
    perror("clock_gettime");
    exit(1);
  }
  return result;
}

// Returns the result of subtracting the timespec 'b' from 'a'.
// The result is normalized so that tv_nsec is between 0 and 999 999 999.
static struct timespec timespec_sub(struct timespec a, struct timespec b) {
  struct timespec result = {.tv_sec = 0, .tv_nsec = 0};
  result.tv_sec = a.tv_sec - b.tv_sec;
  result.tv_nsec = a.tv_nsec - b.tv_nsec;
  while (result.tv_nsec < 0) {
    result.tv_sec -= 1;
    result.tv_nsec += 1000000000;
  }
  return result;
}

// Produces values into segments of a circular buffer, waiting for the consumer
// to finish with the buffer before overwriting it.
//
// The values in the segment are written using the 'store' callback.
__attribute__((always_inline))
static inline void* produce(void* arg, void (*store)(char*, size_t, __m512i)) {
  const struct params* params = arg;
  struct working_set* buffer = &params->mem->buffer;
  const __m512i one = _mm512_set1_epi64(1);
  const size_t iterations = params->iterations;
  const size_t len = params->len;
  __m512i value = _mm512_setzero_epi32();

  pin_to_cpu(params->producer_cpu);
  assert(len > 0 && len % sizeof(value) == 0);

  for (size_t i = 0; i < iterations; i++) {
    const unsigned ix = i & 1;
    char* buf = &buffer->data[ix * len];
    acquire(&buffer->owner[ix], OWNER_PRODUCER);
    store(buf, len, value);
    release(&buffer->owner[ix], OWNER_CONSUMER);
    value = _mm512_add_epi64(value, one);
  }

  return NULL;
}

__attribute__((always_inline))
static inline void store_plain(char* buf, size_t len, __m512i value) {
  const char* end = buf + len;
  for (; buf != end; buf += sizeof(value)) {
    _mm512_store_epi64(buf, value);
  }
}

// Producer that stores data using AVX-512 aligned store instructions.
void* produce_plain(void* arg) {
  return produce(arg, store_plain);
}

__attribute__((always_inline))
static inline void store_cldemote(char* buf, size_t len, __m512i value) {
  const char* end = buf + len;
  for (; buf != end; buf += sizeof(value)) {
    _mm512_store_epi64(buf, value);
    __builtin_ia32_cldemote(buf);
  }
}

// Producer that stores data using AVX-512 aligned store instructions,
// followed by CLDEMOTE to move the data to L3.
void* produce_cldemote(void* arg) {
  return produce(arg, store_cldemote);
}

// Consumes all values emitted by the producer, and stores the benchmark
// results after consuming the last segment.
__attribute__((always_inline))
static inline void* consume(void* arg, void (*sum_stripes)(char*, size_t, __m512i*)) {
  const struct params* params = arg;
  struct timespec start, end;
  struct working_set* buffer = &params->mem->buffer;
  const size_t iterations = params->iterations;
  const size_t len = params->len;

  // Striped summation of the 64-bit values emitted by the producer.
  //
  // The CPU can perform multiple 512-bit load and 512-bit addition
  // operations per cycle, so summing values into a single register
  // could theoretically become a bottleneck.
  __m512i stripe[STRIPES];

  assert(len > 0 && len % sizeof(stripe) == 0);
  pin_to_cpu(params->consumer_cpu);
  start = monotonic_time();

#pragma GCC unroll 8
  for (unsigned j = 0; j < STRIPES; j++) {
    stripe[j] = _mm512_setzero_epi32();
  }

  for (size_t i = 0; i < iterations; i++) {
    const unsigned ix = i & 1;
    char* buf = &buffer->data[ix * len];
    acquire(&buffer->owner[ix], OWNER_CONSUMER);
    sum_stripes(buf, len, stripe);
    release(&buffer->owner[ix], OWNER_PRODUCER);
  }

  __m512i sum = _mm512_setzero_epi32();
#pragma GCC unroll 8
  for (unsigned j = 0; j < STRIPES; j++) {
    sum = _mm512_add_epi64(sum, stripe[j]);
  }

  end = monotonic_time();
  params->mem->result.elapsed = timespec_sub(end, start);
  params->mem->result.sum = sum;
  return NULL;
}

__attribute__((always_inline))
static inline void sum_stripes_plain(char* buf, size_t len, __m512i* stripe) {
  __m512i* ptr = (__m512i*) buf;
  const __m512i* end = ptr + len / sizeof(*ptr);
  for (; ptr != end; ptr += STRIPES) {
#pragma GCC unroll 8
    for (unsigned j = 0; j < STRIPES; j++) {
      __m512i v = _mm512_load_epi64(ptr + j);
      stripe[j] = _mm512_add_epi64(stripe[j], v);
    }
  }
}

// Consumer that loads data using AVX-512 aligned load instructions.
void* consume_plain(void* arg) {
  return consume(arg, sum_stripes_plain);
}

__attribute__((always_inline))
static inline void sum_stripes_cldemote(char* buf, size_t len, __m512i* stripe) {
  __m512i* ptr = (__m512i*) buf;
  const __m512i* end = ptr + len / sizeof(*ptr);
  for (; ptr != end; ptr += STRIPES) {
#pragma GCC unroll 8
    for (unsigned j = 0; j < STRIPES; j++) {
      __m512i v = _mm512_load_epi64(ptr + j);
      stripe[j] = _mm512_add_epi64(stripe[j], v);
      __builtin_ia32_cldemote(ptr + j);
    }
  }
}

// Consumer that loads data using AVX-512 aligned load instructions,
// then removes the loaded cache line from its core-private cache with
// CLDEMOTE.
void* consume_cldemote(void* arg) {
  return consume(arg, sum_stripes_cldemote);
}

// Checks that a completed benchmark run was internally consistent, i.e.
// that it did not skip any values, read any values multiple times, or
// read values before they were updated.
static void check(const struct params* params) {
  unsigned count = sizeof(__m512i) / sizeof(uint64_t);
  char* base = (char*) &params->mem->result.sum;

  // The number of copies of each 512-bit vector in the buffer.
  unsigned long long copies = (params->len / sizeof(__m512i));

  // The first buffer is filled the 64-bit value 0, the second buffer is
  // filled with 1, etc. Therefore, each element of the 512-bit sum is
  // equal to (0 + ... + 0) + (1 + ... + 1) + ... + (iterations - 1 + ...),
  // where there are 'copies' elements in each sub-sum.
  //
  // This simplifies to 'copies * (0 + 1 + ... + iterations - 1)', which
  // simplifies further with the identity '0 + 1 + ... + n == (n + 1) * n / 2'.
  uint64_t expected =
      copies * params->iterations * (params->iterations - 1) / 2;

  for (unsigned i = 0; i < count; i++) {
    uint64_t value;
    memcpy(&value, base + i * sizeof(value), sizeof(value));
    assert(value == expected);
  }
}

// Reports the benchmark results from a completed run.
static void report(const struct params* params) {
  size_t total_bytes = params->len * params->iterations;
  double seconds =
      params->mem->result.elapsed.tv_sec +
      params->mem->result.elapsed.tv_nsec / 1000000000.0;
  printf("Processed %llu bytes (%llu loops × %llu bytes) in %.3f seconds\n",
         (unsigned long long) total_bytes,
         (unsigned long long) params->iterations,
         (unsigned long long) params->len,
         seconds);
  printf("Throughput: %.1f GiB / s\n",
         (double) total_bytes / (1024 * 1024 * 1024) / seconds);
}

// Parses 's' as the benchmark mode -- either "process" or "thread".
static unsigned parse_mode(const char* s) {
  unsigned mode;
  if (strcmp(s, "process") == 0) {
    mode = MODE_PROCESS;
  } else if (strcmp(s, "thread") == 0) {
    mode = MODE_THREAD;
  } else {
    fprintf(stderr, "mode must be \"process\" or \"thread\", not %s\n", s);
    exit(2);
  }
  return mode;
}

// Parses 's' as the hugepage type -- either "explicit" or "transparent".
static unsigned parse_hugepage_type(const char* s) {
  unsigned mode;
  if (strcmp(s, "explicit") == 0) {
    mode = HUGEPAGE_EXPLICIT;
  } else if (strcmp(s, "transparent") == 0) {
    mode = HUGEPAGE_TRANSPARENT;
  } else {
    fprintf(stderr, "hugepage type must be \"explicit\" or \"transparent\", not %s\n", s);
    exit(2);
  }
  return mode;
}

// Parses 's' into a pointer to a producer task implementation.
static task_t parse_producer(const char* s) {
  if (strcmp(s, "plain") == 0) {
    return produce_plain;
  } else if (strcmp(s, "cldemote") == 0) {
    return produce_cldemote;
  } else {
    fprintf(stderr, "producer must be \"plain\" or \"cldemote\", not %s\n", s);
    exit(2);
  }
}

// Parses 's' into a pointer to a consumer task implementation.
static task_t parse_consumer(const char* s) {
  if (strcmp(s, "plain") == 0) {
    return consume_plain;
  } else if (strcmp(s, "cldemote") == 0) {
    return consume_cldemote;
  } else {
    fprintf(stderr, "consumer must be \"plain\" or \"cldemote\", not %s\n", s);
    exit(2);
  }
}

// Parses a CPU ID from 's', and validates that the current process is
// permitted to run on the selected CPU.
static unsigned parse_cpu(const char* s) {
  cpu_set_t mask;
  if (pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask)) {
    perror("pthread_getaffinity_np");
    exit(2);
  }
  char* end;
  unsigned long n = strtoul(s, &end, 0);
  if (n == ULONG_MAX || *end != '\0') {
    fprintf(stderr, "invalid number: %s\n", s);
    exit(2);
  }
  if (n >= CPU_SETSIZE || !CPU_ISSET(n, &mask)) {
    fprintf(stderr, "CPU %s is not in the current process's affinity\n", s);
    exit(2);
  }
  return n;
}

// Parses the human readable size in 's' into a number of bytes.
//
// The size must be either a bare integer number of bytes, or an integer
// suffixed by 'k', 'M', 'G', or 'T'. Suffixes are binary multiples,
// e.g. '4k' is 4096, not 4000.
static size_t parse_size(const char* s) {
  char *end;
  unsigned long long n = strtoull(s, &end, 0);
  if (n == ULLONG_MAX) {
    fprintf(stderr, "number too large: %s\n", s);
    exit(2);
  }
  unsigned long long scale = 1;
  unsigned scale_error = 0;
  switch (*end) {
  case 'T': scale *= 1024;
  case 'G': scale *= 1024;
  case 'M': scale *= 1024;
  case 'k': scale *= 1024;
  case '\0': break;
  default: scale_error = 1;
  }
  if (scale_error || strlen(end) > 1) {
    fprintf(stderr, "invalid suffix: %s\n", end);
    exit(2);
  }
  unsigned long long result;
  if (__builtin_umulll_overflow(n, scale, &result)) {
    fprintf(stderr, "size would overflow: %s\n", s);
    exit(2);
  }
  static_assert(sizeof(result) == sizeof(size_t), "cannot cast result to size_t");
  return result;
}

// Returns '1' if the value of 'n' is a power of two, or '0' otherwise.
static int is_power_of_two(size_t n) {
  return __builtin_popcountll(n) == 1;
}

// Rounds 'x' to the next multiple of 'c' larger than or equal to 'x'.
static unsigned long long round_up(unsigned long long x, unsigned long long c) {
  return ((x - 1) / c + 1) * c;
}

// Creates a shared memory region that contains enough space for a
// 'struct working_set' object that holds two buffers of 'opt->len' bytes.
//
// For simplicity, this function does not track enough information to
// free the memory region. It relies on process exit to clean up.
static union shmem* create_shmem(const struct params* params) {
  int is_default_page_size = params->page_size == sysconf(_SC_PAGESIZE);
  int is_explicit = params->hugepage_type == HUGEPAGE_EXPLICIT;
  int prot = PROT_READ | PROT_WRITE;
  int flags = MAP_ANONYMOUS;
  flags |= (params->mode == MODE_PROCESS) ? MAP_SHARED : MAP_PRIVATE;
  size_t base_size = sizeof(union shmem) + 2 * params->len;
  size_t page_size = params->page_size;
  size_t size = round_up(base_size, page_size);
  if (size > SIZE_MAX / 2) {
    fputs("mmap: Cannot allocate memory\n", stderr);
    exit(1);
  }
  size_t mmap_size;
  if (is_explicit && !is_default_page_size) {
    flags |= MAP_HUGETLB | (__builtin_ctzll(params->page_size) << MAP_HUGE_SHIFT);
    mmap_size = size;
  } else { // HUGEPAGE_TRANSPARENT
    // We can't rely on mmap to align to a hugepage boundary, so we need to
    // over-allocate and then perform the alignment ourselves.
    mmap_size = 2 * size;
  }
  void* base = mmap(NULL, mmap_size, prot, flags, -1, 0);
  if (base == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }
  union shmem* mem =
      is_explicit ? base : (void*) round_up((uintptr_t) base, page_size);
  int advice =
      is_default_page_size || is_explicit ? MADV_NOHUGEPAGE : MADV_HUGEPAGE;
  if (madvise(mem, size, advice)) {
    perror("madvise");
    exit(1);
  }
  // Assign ownership of both buffer segments to the producer.
  for (unsigned i = 0; i < 2; i++) {
    release(&mem->buffer.owner[i], OWNER_PRODUCER);
  }
  return mem;
}

// Runs the producer and consumer in separate threads and waits for them
// to complete.
static void run_threads(struct params* params) {
  pthread_t consumer, producer;
  if (pthread_create(&producer, NULL, params->producer, params)) {
    perror("failed to spawn producer thread");
    exit(1);
  }
  if (pthread_create(&consumer, NULL, params->consumer, params)) {
    perror("failed to spawn consumer thread");
    exit(1);
  }
  if (pthread_join(producer, NULL)) {
    perror("failed to join producer thread");
    exit(1);
  }
  if (pthread_join(consumer, NULL)) {
    perror("failed to join consumer thread");
    exit(1);
  }
}

// Spawns fn(arg) in a child process, and returns the child's process ID.
static int fork_create(pid_t* pid, task_t fn, void* arg) {
  *pid = fork();
  switch (*pid) {
  case -1:
    return -1;
  case 0:
    fn(arg);
    exit(0);
  default:
    return 0;
  }
}

static char* wstatus_to_str(int wstatus) {
    if (WIFEXITED(wstatus))    return "WIFEXITED";
    if (WEXITSTATUS(wstatus))  return "WEXITSTATUS";
    if (WIFSIGNALED(wstatus))  return "WIFSIGNALED";
    if (WTERMSIG(wstatus))     return "WTERMSIG";
    if (WCOREDUMP(wstatus))    return "WCOREDUMP";
    if (WIFSTOPPED(wstatus))   return "WIFSTOPPED";
    if (WSTOPSIG(wstatus))     return "WSTOPSIG";
    if (WIFCONTINUED(wstatus)) return "WIFCONTINUED";
    return "<UNKNOWN WSTATUS>";
}

// Runs the producer and consumer in separate processes and waits for them
// to complete.
static void run_processes(struct params* params) {
  pid_t consumer, producer;
  if (fork_create(&producer, params->producer, params)) {
    perror("failed to spawn producer process");
    exit(1);
  }
  if (fork_create(&consumer, params->consumer, params)) {
    perror("failed to spawn consumer process");
    exit(1);
  }
  for (unsigned i = 0; i < 2; i++) {
    int status;
    pid_t pid = wait(&status);
    if (status) {
      const char* name = pid == consumer ? "consumer" : "producer";
      fprintf(stderr, "%s process failed with error code %d (%s)\n", name, status, wstatus_to_str(status));
      exit(1);
    }
  }
}

static void usage(int exitcode) {
  fputs("Usage: c2c-bench mode hugepage-type producer-type producer-cpu\n", stderr);
  fputs("                 consumer-type consumer-cpu page-size working-set target-size\n", stderr);
  fputs("\n", stderr);
  fputs("  mode           either \"process\" or \"thread\"\n", stderr);
  fputs("  hugepage-type  either \"explicit\" or \"transparent\"\n", stderr);
  fputs("  producer-type  either \"plain\" or \"cldemote\"\n", stderr);
  fputs("  producer-cpu   cpu number to run the producer task on\n", stderr);
  fputs("  consumer-type  either \"plain\" or \"cldemote\"\n", stderr);
  fputs("  consumer-cpu   cpu number to run the producer task on\n", stderr);
  fputs("  page-size      number of bytes per page, e.g. \"2M\"\n", stderr);
  fputs("  working-set    number of bytes in the working set, e.g. 64k\n", stderr);
  fputs("  target-size    target number of bytes to transfer, e.g. 1G\n", stderr);
  exit(exitcode);
}

int main(int argc, const char** argv) {
  if (argc != 10) {
    usage(2);
  }
  struct params params;
  params.mode = parse_mode(argv[1]);
  params.hugepage_type = parse_hugepage_type(argv[2]);
  params.producer = parse_producer(argv[3]);
  params.producer_cpu = parse_cpu(argv[4]);
  params.consumer = parse_consumer(argv[5]);
  params.consumer_cpu = parse_cpu(argv[6]);

  params.page_size = parse_size(argv[7]);
  if (!is_power_of_two(params.page_size)) {
    fprintf(stderr, "page size %s is not a power of two\n", argv[7]);
    exit(2);
  }

  size_t working_set = parse_size(argv[8]);
  size_t step_size = STRIPES * sizeof(__m512i);
  params.len = (working_set - sizeof(union shmem)) / 2 / step_size * step_size;
  if (params.len == 0) {
    fprintf(stderr, "working set size %s is too small\n", argv[8]);
    exit(2);
  }

  size_t total_size = parse_size(argv[9]);
  if (total_size < working_set) {
    fprintf(stderr, "total size %s is smaller than working set\n", argv[9]);
    exit(2);
  }
  params.iterations = total_size / params.len;

  params.mem = create_shmem(&params);

  switch (params.mode) {
  case MODE_PROCESS: run_processes(&params); break;
  case MODE_THREAD: run_threads(&params); break;
  default: assert(0);
  }

  check(&params);
  report(&params);

  return 0;
}
