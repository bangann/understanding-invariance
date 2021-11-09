import torch
from argparse import ArgumentParser
from scn_utils import *

torch.manual_seed(0)

parser = ArgumentParser()
parser.add_argument('--degree', type=int, default=30)
parser.add_argument('--epsilon', type=float, default=3)
parser.add_argument('--data_path', type=str, default='../data/r2n2/images_800_tensor.pt')
parser.add_argument('--data_path_view', type=str, default='../data/r2n2/images_800_tensor_view.pt')
parser.add_argument('--transformation', type=str, choices=['none', 'flip', 'crop', 'rotate', 'cutout', 'view'])
parser.add_argument('--cover_number_method', type=str, choices=['accurate', 'fast'])


def main_view(args):
    # load data
    images_all = torch.load(args.data_path_view).to(device='cuda')  # N * 24 * C* H * W
    print('Data loaded!')

    dis_view = dis_matrix_orbit(images_all).cpu().numpy()
    dis_view = shortest_dist(dis_view)
    print('Finish calculating distance matrix!')

    if args.cover_number_method == 'fast':
        scn = cover_num_fast(dis_view, args.epsilon, dis_view.shape[0]/10)
    else:
        scn = cover_num(dis_view, args.epsilon, dis_view.shape[0])

    return scn


def main(args):
    # load data
    images_all = torch.load(args.data_path).to(device='cuda')  # N * C* H * W
    print('Data loaded!')

    if args.transformation == 'none':
        dis = dis_matrix(images_all, images_all)
    elif args.transformation == 'flip':
        dis = dis_matrix_flip(images_all, 'h')
    elif args.transformation == 'crop':
        dis = dis_matrix_crop(images_all)
    elif args.transformation == 'rotate':
        dis = dis_matrix_rotate(images_all, args.degree)
    elif args.transformation == 'cutout':
        dis = dis_matrix_cutout(images_all)

    print('Finish calculating distance matrix!')

    dis = dis.cpu().numpy()
    dis = shortest_dist(dis)
    if args.cover_number_method == 'fast':
        scn = cover_num_fast(dis, args.epsilon, dis.shape[0]/10)
    else:
        scn = cover_num(dis, args.epsilon, dis.shape[0])

    return scn


if __name__ == "__main__":
    args = parser.parse_args()

    if args.transformation == 'view':
        scn = main_view(args)
    else:
        scn = main(args)

    print('SCN is: ', scn)
