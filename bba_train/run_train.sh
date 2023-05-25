# python3.8 main.py --data_dir datasets/DOTA_devkit/examplesplit/ --batch_size 16 --dataset dota --phase test
# python3.8 main.py --data_dir ./datasets/RSAICP/ --num_epoch 80 --batch_size 16 --dataset rsaicp --phase train
# python3.8 main.py --data_dir ./datasets/RSAICP_608/ --num_epoch 80 --batch_size 16 --dataset rsaicp --resume_train ./weights_rsaicp/model_rsaicp_80.pth --phase train
python main.py --data_dir ./datasets/RSAICP_MS3/ --num_epoch 80 --batch_size 16 --dataset rsaicp  --phase train