db2 connect to test;
time db2 "select PREDICTION from usbankindexed_trans i, table(cachesys.us_bank_predict_udtf(112, i.Index, i.User_id, i.Card, i.Year, i.Month, i.Day, i.Time, i.Amount, i.Use_Chip, i.Merchant_Name, i.Merchant_City, ifnull(i.Merchant_State, ‘CA’), ifnull(i.Zip, '0'), i.MCC, ifnull(i.is_Errors, 'missing_value'), i.is_Fraud)) fetch first 112 rows only"
db2 connect reset;
db2 terminate;
