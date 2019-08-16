#! /bin/bash

echo "-------------- (`date`) Hfdata Job after Close Start --------------"
GIT_PATH=/home/qp/git/pipeline_rq

cd $GIT_PATH
python3 job.daily.close.py

echo "-------------- (`date`) Hfdata Job after Close End --------------"
