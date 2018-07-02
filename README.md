# Using Tensorflow's Estimator and Dataset APIs on popular CNNs

This repo provides boiler plate code to train CNNs using the Tensorflow's popular Estimator and Dataset APIs.
Several popular CNN architectures from different model zoos can be imported, more can added accordingly.

## Features

1. multi-headed outputs
2. multi-gpu usage
3. examples for importing CNNs from tf-slim, tensornets and tensorflow/models/official
4. full process: starting from raw images to making predictions

## Example Usage

```
python main.py \
-root_path /my_images/ \
-model_save_path ./data/model_run \
-model small_cnn \
-max_epoch 10 \
-batch_size 64 \
-image_size 50 \
-num_gpus 0 \
-num_cpus 2 \
-train_fraction 0.8 \
-color_augmentation True
```

## Setup

The code has been tested with Tensorflow 1.8.

## Acknowledgements

Thanks to following model zoos:
- tf-slim models (https://github.com/tensorflow/models/tree/master/research/slim)
- resnet implementation (https://github.com/tensorflow/models/tree/master/official/resnet)
- tensornets (https://github.com/taehoonlee/tensornets)
