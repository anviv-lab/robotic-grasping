#!/usr/bin/env bash

END=9
for ((i=1;i<=END;i++)); do
  curl -O --retry 5 --max-time 15 http://pr.cs.cornell.edu/grasping/rect_data/temp/data0$i.tar.gz
  tar xvzf data0$i.tar.gz
  rm data0$i.tar.gz
done

curl -O --retry 5 --max-time 15 http://pr.cs.cornell.edu/grasping/rect_data/temp/data10.tar.gz
tar xvzf data10.tar.gz
rm data10.tar.gz