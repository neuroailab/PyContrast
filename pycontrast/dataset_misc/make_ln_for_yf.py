import os
from tqdm import tqdm

source_dir = '/data5/chengxuz/Dataset/yfcc/imagenet_size_jpgs_from_tfr'
target_dir = '/data5/chengxuz/Dataset/yfcc/jpgs_in_imagenet_format/train'

os.system('mkdir -p ' + target_dir)

subdirs = os.listdir(source_dir)
for subdir in subdirs:
    curr_subdir = os.path.join(source_dir, subdir)
    all_fldrs = os.listdir(curr_subdir)
    for _fldr in tqdm(all_fldrs):
        new_fldr_name = subdir + '_' + _fldr
        new_fldr_path = os.path.join(target_dir, new_fldr_name)
        old_fldr_path = os.path.join(curr_subdir, _fldr)
        os.system('ln -s %s %s' % (old_fldr_path, new_fldr_path))
