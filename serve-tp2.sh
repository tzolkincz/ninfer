#!/usr/bin/env bash
# drop the runner's single-GPU --device N, add the two-GPU options
args=(); skip=0
for a in "$@"; do
  if [ $skip = 1 ]; then skip=0; continue; fi
  case "$a" in --device) skip=1; continue;; esac
  args+=("$a")
done
exec ./build/apps/ninfer-serve "${args[@]}" --tp 2 --devices 0,1
