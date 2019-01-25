python compress_classifier.py /work/zhang-x2/common/datasets/imagenet-pytorch/ \
    -a resnet50 \
    -b 128 -j 1 --vs 0 \
    --evaluate --pretrained \
    --act-bits 8 --weight-bits 6 \
    --quantize-method ocs \
    --weight-expand-ratio 0.01 \
    --weight-clip-threshold 1.0 \
    --act-clip-threshold 1.0 \
    --profile-batches 4
