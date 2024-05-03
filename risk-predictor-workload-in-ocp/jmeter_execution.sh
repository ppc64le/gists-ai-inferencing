#!/bin/bash
TESTNAME=$1
TESTNAME="power_${TESTNAME}"
USERS=$2
LOOP=500

echo "-----------------------Creating /root/risk_predictor_workload/results/$TESTNAME directory-----------------------"
mkdir /root/risk_predictor_workload/results/$TESTNAME

# Sample collection interval in seconds
NMON_INTERVAL=1

# Number of samples to take. Total run time is $NMON_INTERVAL x $NMON_SAMPLES
# Of course - unless we kill it :)
NMON_SAMPLES=1000

Nodes=( ai-w1.ai.toropsp.com ai-w2.ai.toropsp.com )

echo "-----------------------Start NMON on worker nodes------------------"
for Worker in ${Nodes[*]}
do
   echo "Starting NMON on $Worker"
   ssh -o 'StrictHostKeyChecking no' core@$Worker "nohup ./var/home/core/toolbox-psp/templates/nmon_power_rhel8_16m -ft -s $NMON_INTERVAL -c $NMON_SAMPLES -F test1.nmon /root/cpd48_jmeter/results/$TESTNAME/nmon < /dev/null > std.out 2> std.err &"
done

echo "-----------------------Starting Risk Predictor with $USERS user(s) and $LOOP loop count-----------------------"

"/root/apache-jmeter-5.6.3/bin/jmeter.sh" -n -t /root/OCP-fraud-detection/OCP-fraud-detection-power.jmx -l /root/OCP-fraud-detection/results/$TESTNAME/report-setup.jtl -J"concurrent_users=$USERS" -e -o /root/OCP-fraud-detection/results/$TESTNAME/report-setup

wait
