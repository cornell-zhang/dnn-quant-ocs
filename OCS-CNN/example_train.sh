python compress_classifier.py \
    %DATA_DIR% \
    -a resnet20_cifar \
    --lr 0.1 -p 50 -b 128 -j 1 --epochs 200 \
    --compress=./config/resnet20_cifar_base_fp32.yaml \
    --out-dir="logs_resnet20_cifar_train/" \
    --wd=0.0002 --vs=0
