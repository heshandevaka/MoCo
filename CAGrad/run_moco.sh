mkdir -p logs
dataroot=./cityscapes
weight=equal
seed=0
beta=0.05
delta=0.1

python -u model_segnet_mt.py     --apply_augmentation --dataroot $dataroot --seed $seed --beta $beta --delta $delta --weight $weight --method tracking        > logs/tracking-beta-$beta-delta-$delta-batch-8-$weight-sd$seed.log



