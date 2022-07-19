# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python main.py \
   --output-dir ./output/backbones/ \
   --dataset miniimagenet \
   --data-path ./data/miniimagenet \
   --train-shot 5 \
   --test-shot 5 \
   --name conv/conv_backbone_5W_5K \
   --conv-model conv4 \
   --agg-method mean \
   --random-horizontal-flip \
   --random-resized-crop \
   --color-jitter \
   --warm-up-epochs 100 \
   --step-rate 100 \
   --max-epoch 1000 \
   --test-support-label-noise-list 0.0 0.2 0.4 0.6 0.8
