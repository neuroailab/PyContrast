python main_contrast.py \
  --method InfoMin \
  --cosine \
  --data_folder /data5/chengxuz/Dataset/yfcc/jpgs_in_imagenet_format \
  --multiprocessing-distributed --world-size 1 --rank 0 --arch resnet18 --dist-url tcp://127.0.0.1:23400
