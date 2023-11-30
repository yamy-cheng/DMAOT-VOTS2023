#! /bin/bash

for i in {1..15}
do
    CUDA_VISIBLE_DEVICES=2 vot evaluate --workspace /data2/cym/VOTS2023_Winner  r50_dm_deaot
done


# for i in {1..15}
# do
#     CUDA_VISIBLE_DEVICES=0 vot evaluate --workspace /data2/cym/VOTS2023_Winner  swinb_dm_deaot
# done
