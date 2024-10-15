#!/bin/sh -eu

from="$1"
to="$2"
page_sizes="4k 2M"
sizes="2k 4k 8k 12k 16k 24k 30k 32k 34k 48k 64k 128k 256k 512k 1M 2M 3M 4M 6M 7M 8M 9M 10M 12M 16M 18M 24M 32M 64M 96M 128M"

thp_enabled=/sys/kernel/mm/transparent_hugepage/enabled
thp_shmem_enabled=/sys/kernel/mm/transparent_hugepage/shmem_enabled

if ! grep -Eq '[[](always|madvise)' "${thp_enabled}"; then
  echo >&2 "Transparent huge pages are disabled."
  echo >&2 "Enable them by running:"
  echo >&2
  echo >&2 "  echo madvise | sudo tee ${thp_enabled}"
  echo >&2
  exit 1
fi
if ! grep -Eq '[[](always|within_size|advise)' "${thp_shmem_enabled}"; then
  echo >&2 "Transparent huge pages are disabled for shared mappings."
  echo >&2 "Enable them by running:"
  echo >&2
  echo >&2 "  echo advise | sudo tee ${thp_shmem_enabled}"
  echo >&2
  exit 1
fi

for page_size in ${page_sizes}; do
  for size in ${sizes}; do
    ./c2c-bench process transparent plain "$from" plain "$to" "${page_size}" "${size}" 100G |
      awk -vOFS=, -vpage_size="${page_size}" -vsize="${size}" '
        /^Throughput:/ { print page_size, size, $2 }
      '
  done
done
