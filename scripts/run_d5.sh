#!/bin/bash
# Resumable chunked d=5 proof.  Usage (from anywhere):
#   [START=lo] [ADAPT=1] scripts/run_d5.sh TAU ROOTS CHUNK ENV LENV LISTS FLOOR
# e.g. ADAPT=1 scripts/run_d5.sh 4766 data/roots_gt4766.txt 50 data/S104/merged_env.txt 104 data/lists_F4760.bin 4760
# Writes data/D5_<TAU>/c<lo>.log for tasks [lo, lo+CHUNK); completed chunks are skipped, and the
# script exits if a run for the same TAU is already active. Each pass continues from the end of
# the previous chunk, so chunks of different sizes stay contiguous (START sets where a pass begins).
# With ADAPT=1 the chunk size doubles after a chunk under 2 min and halves after one over 20 min
# (between 25 and 2000 tasks): startup (~21 s) stays cheap and a stopped job loses little.
# Check with scripts/check_d5.py data/D5_<TAU>.
TAU=$1; ROOTS=$2; CH=$3; ENV=$4; LENV=$5; LISTS=$6; FL=$7; START=${START:-0}
cd "$(dirname "$0")/.." || exit 1
if pgrep -f "bb5R data $TAU " >/dev/null; then echo "already running"; exit 0; fi
D=data/D5_$TAU; mkdir -p $D
total=$(TLO=0 THI=0 DSYM=12 bin/bb5R data $TAU $ROOTS $D/probe 3 0 $ENV $LENV $LISTS $FL 2>/dev/null | grep -oP '^total tasks generated: \K[0-9]+')
echo "total tasks: $total"
lo=$START
while ((lo < total)); do
  hi=$((lo+CH)); out=$D/c$lo
  if ! grep -q "^done" $out.log 2>/dev/null; then
    TLO=$lo THI=$hi DSYM=12 bin/bb5R data $TAU $ROOTS $out 3 0 $ENV $LENV $LISTS $FL > $out.log 2>&1
  fi
  if ! grep -q "^done" $out.log; then echo "FAILED $out.log"; exit 1; fi
  c=$(grep -oP 'CHUNK tasks \[\d+,\K\d+' $out.log); lo=${c:-$hi}
  if [ -n "$ADAPT" ]; then
    secs=$(grep -oP '^done in \K[0-9]+' $out.log); secs=${secs:-0}
    if ((secs < 120 && CH < 2000)); then CH=$((CH*2 > 2000 ? 2000 : CH*2)); elif ((secs > 1200 && CH > 25)); then CH=$((CH/2)); fi
  fi
done
echo ALLDONE
