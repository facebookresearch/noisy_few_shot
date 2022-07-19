# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python main.py \
   --dataset miniimagenet \
   --data-path ./data/miniimagenet \
   --output-dir ./output/tranfs/ \
   --name trans3_conv4_5w_5k \
   --conv-model conv4 \
   --transformer-layers 3 \
   --trans-d-model 128 \
   --ortho-proj \
   --cls-type rand_const \
   --output-from-cls \
   --binary-outlier-loss-weight 0.5 \
   --clean-proto-loss-weight 1.0 \
   --agg-method mean \
   --random-horizontal-flip \
   --random-resized-crop \
   --lr 0.0005 \
   --warm-up-epochs 100 \
   --step-rate 250 \
   --step-gamma 0.7 \
   --max-epoch 2000 \
   --noise-type sym_swap \
   --train-support-label-noise-choices 0.2 0.4 \
   --test-support-label-noise-list 0.0 0.2 0.4 0.6 0.8 \
   --load-checkpoint-path ./output/backbones/conv/conv_backbone_5W_5K/checkpoint.pth \
   --freeze-conv
