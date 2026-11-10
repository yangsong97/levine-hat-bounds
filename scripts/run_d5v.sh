#!/bin/bash
# Resumable d=5 proof with the coupled-bound engine bb5V. Exits if a run for this TAU is already active.
# Usage (from anywhere): [START=lo] scripts/run_d5v.sh TAU ROOTS CH ENV LENV LISTS FLOOR
# Writes data/Z5_<TAU>/c<lo>.log for tasks [lo, lo+CH). Chunk size adapts between 25 and 4000 tasks
# (doubles after a chunk under 2 min, halves after one over 20 min); chunks stay contiguous.
TAU=$1; ROOTS=$2; CH=$3; ENV=$4; LENV=$5; LISTS=$6; FL=$7; START=${START:-0}
cd "$(dirname "$0")/.." || exit 1
if pgrep -f "bb5V data $TAU " >/dev/null; then echo "already running"; exit 0; fi
D=data/Z5_$TAU; mkdir -p $D
total=$(TLO=0 THI=0 DSYM=12 bin/bb5V data $TAU $ROOTS $D/probe 3 0 $ENV $LENV $LISTS $FL 2>/dev/null | grep -oP '^total tasks generated: \K[0-9]+')
echo "total tasks: $total"
lo=$START
while ((lo < total)); do
  hi=$((lo+CH)); out=$D/c$lo
  if ! grep -q "^done" $out.log 2>/dev/null; then
    TLO=$lo THI=$hi DSYM=12 bin/bb5V data $TAU $ROOTS $out 3 0 $ENV $LENV $LISTS $FL > $out.log 2>&1
  fi
  if ! grep -q "^done" $out.log; then echo "FAILED $out.log"; exit 1; fi
  c=$(grep -oP 'CHUNK tasks \[\d+,\K\d+' $out.log); lo=${c:-$hi}
  secs=$(grep -oP '^done in \K[0-9]+' $out.log); secs=${secs:-0}
  if ((secs < 120 && CH < 4000)); then CH=$((CH*2 > 4000 ? 4000 : CH*2)); elif ((secs > 1200 && CH > 25)); then CH=$((CH/2)); fi
done
echo ALLDONE
