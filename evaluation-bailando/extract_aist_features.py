import os
import numpy as np
import argparse

from utils.features.kinetic import extract_kinetic_features
from utils.features.manual_new import extract_manual_features

import multiprocessing
import functools


parser = argparse.ArgumentParser(
    description='')
parser.add_argument(
    '--save_dir',
    type=str,
    # default='eval-data/eval-features/',
    default='eval-data/freeze0',
    # default='eval-data/gt-features/',
    help='output local dictionary that stores features.')
parser.add_argument(
    '--motion_dir',
    type=str,
    # default="eval-data/model-output",
    # default="eval-data/val-gt",
    default="/home/s3007294/motion-diffusion-model/save/my_swdance/samples/freeze0_validation/motions",
    # default="/home/s3007294/motion-diffusion-model/save/my_swdance/samples/swdance_val_just_texts/motions",
    help="Directory with ground motions to process"
)
FLAGS = parser.parse_args()
    

def main(file_name, motion_dir):
    seq_name = file_name[:-4]

    man_file = os.path.join(FLAGS.save_dir, 'manual_features_new', seq_name+"_manual.npy")
    kin_file = os.path.join(FLAGS.save_dir, 'kinetic_features', seq_name+"_kinetic.npy")
    
    keypoints3d = np.load(os.path.join(motion_dir, file_name)) 

    if not os.path.exists(man_file):
        features = extract_manual_features(keypoints3d)
        np.save(man_file, features)
    else:
        print(f'man file of {seq_name} exists')

    if not os.path.exists(kin_file):
        features = extract_kinetic_features(keypoints3d)
        np.save(kin_file, features)
    else:
        print(f'kin file of {seq_name} exists')

    print (seq_name, "is done")


if __name__ == '__main__':
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'kinetic_features'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.save_dir, 'manual_features_new'), exist_ok=True)
    
    file_names = [f for f in os.listdir(FLAGS.motion_dir) if not f.startswith("M")]
    print(f'motion dir: {FLAGS.motion_dir}, files: {file_names}')

    # processing
    process = functools.partial(main, motion_dir=FLAGS.motion_dir)
    pool = multiprocessing.Pool(12)
    pool.map(process, file_names)