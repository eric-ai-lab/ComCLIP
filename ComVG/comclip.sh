#!/bin/bash
# $1 image_folder
# $2 grit densecaption result
# $3 grit model path
# $4 data_csv path
# $5 openai API key
# $6 vision encoder vision
mkdir $2
python GRiT/demo.py --test-task DenseCap --config-file GRiT/configs/GRiT_B_DenseCap_ObjectDet.yaml  --input $1 --output $2 --opts MODEL.WEIGHTS $3
mkdir ComVG/matched_relation
python ComVG/match_relation.py --densecaption_path $2 --dataset_path $4 --openai $5
python ComVG/comclip.py --dataset_path $4 --image_path $1 --densecaption_path $2 --model $6