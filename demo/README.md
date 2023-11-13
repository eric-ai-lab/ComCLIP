# ComCLIP

ComCLIP: Training-Free Compositional Image and Text Matching

ComVG & SVO_Probes:
1. baseline.ipynb: baseline clip running on datasets
2. ComCLIP.ipynb: main algorithm for ComCLIP
3. OpenCLIP.ipynb: main algorithm for ComCLIP on openclip
4. parse_image.py: helper functions to create subimages.
5. match_relation.ipynb: gpt prompt to match dense captions to subject, object, predicates

flickr30k_mscoco:
1. CLIP_ComCLIP.ipynb: comclip and clip retrieval on both datasets
2. parse_image.py: helper functions to create subimages.
3. parse_relation.ipynb: gpt prompt to parse subject, object, predicates and their connection in text
4. match_relation.ipynb: gpt prompt to match dense captions to subject, object, predicates

VL-checklist:
1. ComBLIP_BLIP.ipynb: main algorithm for ComBLIP, BLIP2 baseline
2. ComCLIP_CLIP.ipynb: main algorithm for ComCLIP, CLIP baseline
3. parse_image.py: helper functions to create subimages
4. parse_relation.ipynb: gpt prompt to parse subject, object, predicates and their connection in text
5. match_relation.ipynb: gpt prompt to match dense captions to subject, object, predicates

winoground:
1. ComBLIP_BLIP.ipynb: main algorithm for ComBLIP, BLIP2 baseline on winoground
2. ComCLIP_CLIP.ipynb: main algorithm for ComCLIP, CLIP baseline on winoground
3. parse_image.py: helper functions to create subimages
4. parse_relation.ipynb: gpt prompt to parse subject, object, predicates and their connection in text
5. match_relation.ipynb: gpt prompt to match dense captions to subject, object, predicates
