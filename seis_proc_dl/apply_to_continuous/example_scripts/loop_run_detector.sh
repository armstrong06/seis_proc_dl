#!/bin/bash

year=2022
month=9
day=5
n_days=1

while read -r stat chan; do 
    echo $stat $chan
    outfile="logs/out.${stat}.${chan}"
    echo $outfile
    nohup python -u run_detector.py -s $stat -c $chan  -y $year -m $month -d $day -n $n_days &> $outfile & 
done <<< "$(ls ../downloaded_all_data/2022/09/05/ | awk 'BEGIN{FS="[.|__]"} {if($4=="HHZ")print $2, substr($4, 1, length($4)-1)}' | sort | uniq)"
#done <<< "$(ls ../downloaded_all_data/2022/09/05/ | awk 'BEGIN{FS="[.|__]"} {if($4=="HHZ")print $2, $4}' | sort | uniq)"
#done <<< "$(ls downloaded_all_data/2022/09/05/ | awk 'BEGIN{FS="[.|__]"} {print $2, $4}' | sed 's/.$//' | sort | uniq)"
