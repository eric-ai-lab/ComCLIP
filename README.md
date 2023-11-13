## ComCLIP: Training-Free Compositional Image and Text Matching

This is the code implementation for the paper titled: "ComCLIP: Training-Free Compositional Image and Text Matching" [[Arxiv](https://arxiv.org/abs/2211.13854)]

<div align=center>  
<img src='.github/overview.png' width="80%">
</div>

## Todo
- [ ] Release dataset
- [x] Release code
- [ ] Release playground
- [ ] HuggingFace and Kaggle dataset
- [ ] Website 
- [ ] Readme update 


## Datasets
Please follow the instructions below to prepare the datasets.
1. Winoground <br/>
[Download images](https://huggingface.co/datasets/facebook/winoground/tree/main/data) and store them as `datasets/winoground_images`. Code includes the download of csv file.
2. Compositional Visual Genome (ComVG) <br/>
[Download images](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) and store them as `datasets/comvg_images`. Test csv file at at `datasets/ComVG.csv`
3. SVO-Probe<br/>
[Download dataset](https://github.com/google-deepmind/svo_probes/blob/main/svo_probes.csv) and store the images as `datasets/SVO-Probes`. Store csv as `datasets/svo-probes.csv`
4. Flickr30k<br/>
[Download images](https://shannon.cs.illinois.edu/DenotationGraph/) and store them as `datasets/flickr30k_image`. Test pickle file is `datasets/flickr30k_test.pkl`

## Usage 
### Preparation 
Please follow [GRiT Setup](https://github.com/JialianW/GRiT/blob/master/docs/INSTALL.md) and [CLIP Setup](https://github.com/openai/CLIP/tree/main) first.
<pre>conda create --name comclip python=3.10
conda activate comclip
pip install -r requirements.txt
</pre>

### Winoground
<pre>cd winogroud
### clip baseline
python clip_baseline.py --huggingface_token HUGGINGFACE_TOKEN
### blip baseline
python blip_baseline.py --huggingface_token HUGGINGFACE_TOKEN

### comclip 
comclip.sh IMAGE_PATH DENSE_CAPTION_PATH PARSE_TEXT_PATH GRiT_MODEL HUGGINGFACE_KEY OPENAI_KEY
### comblip
comclip.sh IMAGE_PATH DENSE_CAPTION_PATH PARSE_TEXT_PATH GRiT_MODEL HUGGINGFACE_KEY OPENAI_KEY
</pre>

### ComVG & SVO-Probes
<pre>
cd ComVG
### clip baseline
python clip_baseline.py --model VISION_ENCODER_TYPE --data_path CSV_PATH
### comclip 
comclip.sh
</pre>

### Flick30k (image retrival)
<pre>
cd image_retrival
### clip baseline
python clip_baseline.py --model VISION_ENCODER_TYPE --data_path CSV_PATH
### comclip 
comclip.sh
</pre>

## Acknowledgement 
This repo has inherited 2 open source projects: 1.[GRiT](https://github.com/JialianW/GRiT) 2.[CLIP](https://github.com/openai/CLIP) <br/>
We thank the authors for their amazing work.

## Citation