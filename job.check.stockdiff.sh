#! /bin/bash

echo "-------------- (`date`) Hfdata Job after Close Start --------------"
GIT_PATH=/home/qp/git/pipeline_rq

cd $GIT_PATH
python3 job.check.stockdiff.py

echo "-------------- (`date`) Hfdata Job after Close End --------------"
