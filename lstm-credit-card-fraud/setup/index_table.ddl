drop table usbank.indexed_trans;

create table usbank.indexed_trans (
index int,
User_id int,
Card int,
Year int,
Month int,
Day int,
Time char(5),
Amount varchar(20),
Use_Chip varchar(128),
Merchant_Name varchar(128),
Merchant_City varchar(128),
Merchant_State varchar(128),
Zip double,
MCC int,
is_Errors varchar(128),
Is_Fraud varchar(4) );
