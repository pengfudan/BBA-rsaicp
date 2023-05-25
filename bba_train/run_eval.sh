# python3.8 main.py --data_dir datasets/DOTA_devkit/examplesplit/ --batch_size 16 --dataset dota --phase test
# python3.8 main.py --data_dir ./datasets/RSAICP/ --num_epoch 80 --batch_size 16 --dataset rsaicp --phase train
python3.8 main.py --data_dir ./datasets/RSAICP/ --conf_thresh 0.31 --batch_size 16 --dataset rsaicp --phase eval