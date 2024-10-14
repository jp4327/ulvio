import argparse
import os
import torch
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
from collections import defaultdict
from utils.kitti_eval_temporal import KITTI_tester
import numpy as np
import math

from ULVIO import ULVIO
from params import par
from utils.utils import table_calculate_size

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='./data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')
parser.add_argument('--experiment_name', type=str, default='test', help='experiment name')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length')

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['05', '07', '10'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')

parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--model', type=str, default='./model_ulvio_v1.pth', help='path to the pretrained model')

args = parser.parse_args()

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def main():
    # GPU selections
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])

    # Initialize the tester
    tester = KITTI_tester(args)
    model = ULVIO(par)

    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    # model.load_state_dict(torch.load(par.load_ckpt, map_location='cpu'))
    print('load model %s'%args.model)

    # Feed model to GPU
    model.cuda(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids = gpu_ids)
    model.eval()

    # table_calculate_size(model) # calculate model size

    with torch.no_grad(): 
        model.eval()
        errors = tester.eval(model, num_gpu=len(gpu_ids))

    for i, seq in enumerate(args.val_seq):
        message = f"Seq: {seq}, t_rel: {tester.errors[i]['t_rel']:.4f}, r_rel: {tester.errors[i]['r_rel']:.4f}, "
        message += f"t_rmse: {tester.errors[i]['t_rmse']:.4f}, r_rmse: {tester.errors[i]['r_rmse']:.4f}, "
        print(message)

    # # Create Dir
    # result_dir = Path(args.save_dir)
    # result_dir.mkdir_p()
    # experiment_dir = result_dir.joinpath('{}/'.format(args.experiment_name))
    # experiment_dir.mkdir_p()

    # tester.generate_plots(experiment_dir, 30) # generate plots 

if __name__ == "__main__":
    main()