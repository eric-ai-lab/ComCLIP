#!/bin/bash

cd ../grit 
"input path $1"
"Dense caption path $2"
"Relation path $3"
"model path $4"
"huggingface key $5"
"openai key $6"
"image path $7"
python demo.py --test-task DenseCap --config-file configs/GRiT_B_DenseCap_ObjectDet.yaml  --input $1 --output $2 --opts $4
cd ../ComVG
mkdir matched_relation
python match_relation.py --DenseCaptionPath $2 --RelationPath $3 --openai $6
python comblip.py --RelationPath $3 --image_path $7

