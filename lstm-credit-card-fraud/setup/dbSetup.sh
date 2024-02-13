# complet prereq steps here: https://www.ibm.com/docs/en/db2/11.1?topic=servers-prerequisites-db2-database-server-installation-linux-unix, and command examples:  https://www.ibm.com/docs/en/db2/11.1?topic=unix-creating-group-user-ids-db2-database-installation
# create db2inst1 user then db2 instance under the db2 install dir.  e.g.
#    groupadd -g 969 db2iadm1
#    groupadd -g 968 db2fsdm1
#    groupadd -g 967 dasadm1
# useradd -u 1004 -g db2iadm1 -m -d /home/db2inst1 db2inst1
# useradd -u 1003 -g db2fsdm1 -m -d /home/db2fenc1 db2fenc1
# useradd -u 1002 -g dasadm1 -m -d /home/dasusr1 dasusr1
# passwd db2inst1; passwd db2fenc1; passwd dasusr1
#
# cd /opt/ibm/db2/V11.5/instance
# ./db2icrt -s ese -u db2fenc1 db2inst1

db2set DB2_4K_DEVICE_SUPPORT=YES
db2set DB2COMM=TCPIP
db2start
db2 create db test
db2 connect to test
db2 update db cfg for test using LOGFILSIZ 51200 LOGPRIMARY 130 LOGSECOND 120
db2 connect reset
db2 terminate
db2stop
db2start

