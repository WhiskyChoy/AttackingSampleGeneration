import os


def ensure_pre_dirs_exists(*input_dirs):
    for dir_item in input_dirs:
        ensure_pre_dir_exists(dir_item)


def ensure_pre_dir_exists(input_dir):
    pre_dir = '/'.join(input_dir.split('/')[:-1])
    if not (os.path.exists(pre_dir) and os.path.isdir(pre_dir)):
        os.makedirs(pre_dir)
