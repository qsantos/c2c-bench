#!/bin/sh -eu

if [ "$#" -ne 3 ]; then
  echo >&2 "Usage: $0 microarchitecture-name producer-cpu-name consumer-cpu-name"
  echo >&2
  echo >&2 "Example: $0 Zen4c 'Core 0, Thread 0' 'Core 0, Thread 1'"
  exit 1
fi

cpu="$1"
from="$2"
to="$3"
tempfile="$(mktemp)"
awk -vFS=, -vOFS=, '
  {
    page_sizes[$1] = 1
    working_sets[$2] = 1
    throughput[$1,$2] = $3
  }
  function bytes(s) {
    if (s ~ "[0-9]$") { return s }
    if (s ~ "k$") { return s * 1024 }
    if (s ~ "M$") { return s * 1024 * 1024 }
    if (s ~ "G$") { return s * 1024 * 1024 * 1024 }
    return "badsize:" s
  }
  END {
    printf "%s", "Working Set Size"
    for (p in page_sizes) {
      printf ",%s", ("Page Size: " p)
    }
    printf "\n"
    for (w in working_sets) {
      printf "%d", bytes(w)
      for (p in page_sizes) {
        printf ",%d", bytes(throughput[p,w])
      }
      printf "\n"
    }
  }
' >"${tempfile}"
gnuplot -e "
  set datafile separator ',';
  set logscale x 2;
  set xtics 2 font ',9';
  set ytics font ',9';
  set grid back;
  set format x \"%.0b\n%BB\";
  set xlabel 'Working Set Size';
  set ylabel 'Throughput (GiB / s)';
  set key autotitle columnhead font ',9' left;
  set title \"$cpu Core-to-core Throughput\\n{/*0.8 $from ⇒ $to}\" font ',14';
  set terminal pngcairo size 700,400;
  set output '/dev/stdout';
  plot '${tempfile}' using 1:2 with lines smooth unique, \
       '' using 1:3 with lines smooth unique;
"
rm -f "${tempfile}"
