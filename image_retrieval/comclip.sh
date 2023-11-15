#!/bin/bash
# $1 image_folder
# $2 grit densecaption result
# $3 grit model path
# $4 data_csv path
# $5 openai API key
# $6 vision encoder vision
mkdir $2
python GRiT/demo.py --test-task DenseCap --config-file GRiT/configs/GRiT_B_DenseCap_ObjectDet.yaml  --input $1 --output $2 --opts MODEL.WEIGHTS $3
mkdir image_retrieval/relation_json
python image_retrieval/parse_relation.py --relation_json_path image_retrieval/relation_json --data_path $4
mkdir image_retrieval/matched_relation
python image_retrieval/match_relation.py --densecaption_path $2 --dataset_path $4 --openai $5
python image_retrieval/comclip.py --dataset $4 --image_path $1 --text_relation_path image_retrieval/relation_json --densecaption_path $2 --model $6
