#!/bin/bash

python visualization/extract_object_views.py \
  --frames-dir intermediate_data/step1_frames/full_fast_train_109\
  --output-dir output_viz/full_fast_train_109_object_views \
  --config-file configs/custom/videomt_custom_ViTL.yaml \
  --confidence-threshold 0.7 \
  --min-views 5 \
  --top-n 5 \
  --max-views 8 \
  --opts MODEL.WEIGHTS output_videomt_custom_ViTL/model_final.pth
