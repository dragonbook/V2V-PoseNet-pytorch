# V2V-PoseNet-pytorch


## Warning
Need to disable cudnn for batchnorm, or just only use cuda instead. With cudnn for batchnorm and in float precision, the model cannot train well. My simple experiments show that:

```
cudnn+float: Not work 
cudnn+double: work, bug slow
cudnn+float+(disable batchnorm's cudnn): work
cuda+(float/double): work, but uses much more memroy
```

There is a similar issue pointed out by https://github.com/Microsoft/human-pose-estimation.pytorch. As suggested, disable cudnn for batchnorm:

```
PYTORCH=/path/to/pytorch
for pytorch v0.4.0
sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
for pytorch v0.4.1
sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
```

