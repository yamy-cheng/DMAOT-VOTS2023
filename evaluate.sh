#! /bin/bash

for i in {1..15}
do
    CUDA_VISIBLE_DEVICES=0 vot evaluate --workspace /disk2/shared_dataset/CYM/VOTS2023_Winner  r50_deaot
done

vot pack --workspace /disk2/shared_dataset/CYM/VOTS2023_Winner  r50_deaot
# for i in {1..15}
# do
#     CUDA_VISIBLE_DEVICES=0 vot evaluate --workspace /data2/cym/VOTS2023_Winner  swinb_dm_deaot
# done
