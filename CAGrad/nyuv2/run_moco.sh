mkdir -p logs
dataroot=./nyuv2
weight=equal
seed=0
beta=0.99
delta=0.1
sigma2=0.0
sigma3=0.0
smgdc_method=raw_loss

python -u model_segnet_mt.py     --apply_augmentation --dataroot $dataroot --seed $seed --beta $beta --delta $delta --sigma2 $sigma2 --sigma3 $sigma3 --weight $weight --smgdc_method $smgdc_method --method tracking > logs/tracking-smgdc_method-$smgdc_method-beta-$beta-delta-$delta-sigma2-$sigma2-sigma3-$sigma3-batch-2-$weight-sd$seed.log


