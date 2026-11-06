#!/bin/bash
# Resumable chunked d=4 run.  Usage (from anywhere):  PROG=bb4y DSYM=8 HIGHF=4760 scripts/run_chunks.sh L CHUNK
# Reads data/roots_317plus.txt and the frame tables in data/; writes data/S<L>/c<lo>.{log,_env.txt,_high.txt}.
# Completed chunks (log ends with "done") are skipped, so rerunning resumes.
L=$1; CH=$2
cd "$(dirname "$0")/.." || exit 1
P=bin/${PROG:-bb4s}
mkdir -p data/S$L
total=$($P data $L data/roots_317plus.txt data/S$L/probe 4 0 0 2>/dev/null | grep -oP '^tasks: \K[0-9]+')
echo "total tasks: $total"
for ((lo=0; lo<total; lo+=CH)); do
  hi=$((lo+CH)); out=data/S$L/c$lo
  if grep -q "^done" $out.log 2>/dev/null; then continue; fi
  $P data $L data/roots_317plus.txt $out 4 $lo $hi > $out.log 2>&1
done
echo ALLDONE
