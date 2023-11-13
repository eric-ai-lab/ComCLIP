#!/bin/bash

cd grit 
"image folder input path $1"
"Dense caption path $2"
"Relation path $3"
"model path $4"
"huggingface key $5"
"openai key $6"
python demo.py --test-task DenseCap --config-file configs/GRiT_B_DenseCap_ObjectDet.yaml  --input $1 --output $2 --opts $4
cd ../winoground
python parse_relation.py --OutputRelationPath $3 --huggingface $5 --openai $6
mkdir matched_relation
python match_relation.py --DenseCaptionPath $2 --RelationPath $3 --openai $6
python comclip.py --RelationPath $3

