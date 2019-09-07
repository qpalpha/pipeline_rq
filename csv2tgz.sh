#!/bin/bash
DIR=$1

cd $DIR
files=$(ls|grep csv)

for csv in $files
do
    tgz=${csv/csv/tgz}
    tar zcPf $tgz $csv
    rm $csv
done
