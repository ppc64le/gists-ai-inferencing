#!/bin/bash
  
set -x

db2 connect to test
db2 "import from $1 of del modified by coldel, messages msgs.txt insert into $2"

db2 connect reset
set +x

