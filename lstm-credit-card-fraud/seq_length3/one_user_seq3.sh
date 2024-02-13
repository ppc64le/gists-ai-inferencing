#!/bin/bash

db2 connect to test;

#Print the system date and time in seconds since 1970-01-01 00:00:00 UTC
startTime=$(date +%s);
 
db2 "select IDX, PREDICTION, ACTUAL from usbank.indexed_trans_seq3 i, table(cachesys.us_bank_predict_udtf_seq3(19200, i.Index, i.User_id, i.Card, i.Year, i.Month, i.Day, i.Time, i.Amount, i.Use_Chip, i.Merchant_Name, i.Merchant_City, ifnull(i.Merchant_State, 'CA'), ifnull(i.Zip, '0'), i.MCC, ifnull(i.is_Errors, 'missing_value'), i.is_Fraud)) where i.index<=676137"

endTime=$(date +%s);

#Subtract endTime from startTime to get the total execution time
totalTime=$(($endTime-$startTime));

echo "realtime: $totalTime";

db2 connect reset;
db2 terminate;
