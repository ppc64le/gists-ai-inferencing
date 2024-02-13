#!/bin/sh

set -x
# Sample collection interval in seconds
NMON_INTERVAL=1

# Number of samples to take.  Total run time is $NMON_INTERVAL x $NMON_SAMPLES
# Of course - unless we kill it :)
NMON_SAMPLES=100

i=( 1 )

nmon -ft -s $NMON_INTERVAL -c $NMON_SAMPLES
for j in ${i[*]}
do
   for ((k = 1; k <= j; k++))
   do
      ./one_user.sh 2>&1 | tee output${j}.txt &
   done
done

set +x
