#!/bin/bash

"image folder input path $1"
"Dense caption path $2"
"Relation path $3" 
"model path $4" 
"huggingface key $5"
"openai key $6"

mkdir $2
mkdir $3
python GRiT/demo.py --test-task DenseCap --config-file GRiT/configs/GRiT_B_DenseCap_ObjectDet.yaml  --input $1 --output $2 --opts MODEL.WEIGHTS $4
python winoground/parse_relation.py --output_relation_path $3 --huggingface $5 --openai $6
mkdir winoground/matched_relation
python winoground/match_relation.py --huggingface_token $5 --DenseCaptionPath $2 --RelationPath $3 --openai $6
python winoground/comclip.py --RelationPath $3 --CaptionPath $2 --image_path $1 --huggingface_token $5

