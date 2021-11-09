"""
Contains 4 kinds of functions:
- transformations
- distribution matrix calculation for all images (a tensor with size N x C x H x W)
- distribution matrix calculation for all orbits (a tensor with size N x O x C x H x W)
- covering number calculation

Supported transformations:
- None
- Flip
- Crop
- Rotate
- Cutout
- ColorJitter

"""

import torch
import numpy as np
from sklearn_extra.cluster import KMedoids
from scipy.sparse.csgraph import shortest_path
import torchvision.transforms as transforms


def dis_matrix(images1, images2):
    """Euclidean Distance between every two images
    :param images1: tensor with size N x C x H x W
    :param images2: same size as images1
    :return: distance matrix, tensor N x N
    """
    return torch.cdist(torch.flatten(images1, start_dim=1), torch.flatten(images2, start_dim=1), p=2)


def dis_matrix_orbit(images):
    """Shortest Distance between every two images when considering data transformations. Distance matrix is calculated based on orbits
    :param images: tensor with size N x O x C x H x W
    :return: distance matrix, tensor N x N
    """
    orbit_len = images.shape[1]
    N = images.shape[0]
    dist_mat = ((torch.ones(N, N) - torch.eye(N)) * 1000.0).cuda()

    for i in range(orbit_len):
        for j in range(orbit_len):
            dist_new = dis_matrix(images[:, i, :], images[:, j, :]).cuda()
            dist_mat = torch.min(dist_mat, dist_new)
    return dist_mat


def dis_matrix_flip(images, flip_type='h'):
    """ Flipping
    :param images: tensor with size N x C x H x W
    :param flip_type: 'h', 'v', 'hv'
    :return: distance matrix, tensor N x N
    """
    dist_mat = dis_matrix(images, images)

    if flip_type == 'h' or flip_type == 'hv':
        im_flip_h = transforms.RandomHorizontalFlip(1)(images)
        dis_new = dis_matrix(images, im_flip_h)
        dist_mat = torch.min(dist_mat, dis_new)
    if flip_type == 'v' or flip_type == 'hv':
        im_flip_v = transforms.RandomVerticalFlip(1)(images)
        dis_new = dis_matrix(images, im_flip_v)
        dist_mat = torch.min(dist_mat, dis_new)
    if flip_type == 'hv':
        dis_new = dis_matrix(im_flip_h, im_flip_v)
        dist_mat = torch.min(dist_mat, dis_new)
    return dist_mat


def crop_images(images, repeat):
    """
    :param images: tensor with size N x C x H x W
    :param repeat:
    :return: tensor with size N x repeat x C x H x W
    """
    images_crop = []
    resize = images.shape[-1]
    transforms_crop = transforms.RandomCrop(resize, padding=4)
    images_crop.append(images.unsqueeze(1))
    for _ in range(repeat-1):
        images_crop.append(transforms_crop(images).unsqueeze(1))
    images_crop = torch.cat(images_crop, dim=1)
    return images_crop


def dis_matrix_crop(images, repeat=50):
    """ Cropping
    :param images: tensor with size N x C x H x W
    :param repeat: repeat times to approximate the orbit
    :return: distance matrix, tensor N x N
    """
    images = crop_images(images, repeat)
    dist_mat = dis_matrix_orbit(images)
    return dist_mat


def rotate_images(images, deg):
    """
    :param images: tensor with size N x C x H x W
    :param deg: degree
    :return: tensor with size N x O x C x H x W
    """
    images_rotate = []
    for k in range(-deg, deg + 1, 2):
        images_r = transforms.RandomRotation((k, k))(images)
        images_rotate.append(images_r.unsqueeze(1))
    images_rotate = torch.cat(images_rotate, dim=1)
    return images_rotate


def dis_matrix_rotate(images, deg=30):
    """ Rotation
    :param images: tensor with size N x C x H x W
    :param deg: degree (int)
    :return: distance matrix, tensor N x N
    """
    images_rotate = rotate_images(images, deg)
    dist_mat = dis_matrix_orbit(images_rotate)
    return dist_mat


def cutout_images(images, repeat):
    """
    :param images: tensor with size N x C x H x W
    :param repeat:
    :return: tensor with size N x repeat x C x H x W
    """
    images_cutout = []
    transforms_cutout = transforms.RandomErasing(p=1, value=0.5, scale=(0.05, 0.05), ratio=(1,1))  # default scale=(0.02, 0.33), ratio=(0.3, 3.3)
    images_cutout.append(images.unsqueeze(1))
    for _ in range(repeat-1):
        images_cutout.append(transforms_cutout(images).unsqueeze(1))
    images_cutout = torch.cat(images_cutout, dim=1)
    return images_cutout


def dis_matrix_cutout(images, repeat=50):
    """ Cutout
    :param images: tensor with size N x C x H x W
    :param repeat: repeat times to approximate the orbit
    :return: distance matrix, tensor N x N
    """
    images = cutout_images(images, repeat)
    dist_mat = dis_matrix_orbit(images)
    return dist_mat


def jitter_img(images, repeat):
    """
    :param images: tensor with size N x C x H x W
    :param repeat:
    :return: tensor with size N x repeat x C x H x W
    """
    images_jitter = []
    transforms_jitter = transforms.ColorJitter(.25, .25, .25)
    images_jitter.append(images.unsqueeze(1))

    for i in range(repeat-1):
        images_jitter.append(transforms_jitter(images).unsqueeze(1))
    images_jitter = torch.cat(images_jitter, dim=1)
    return images_jitter


def dis_matrix_jitter(images, repeat=50):
    """ ColorJitter
    :param images: tensor with size N x C x H x W
    :param repeat: repeat times to approximate the orbit
    :return: distance matrix, tensor N x N
    """
    images_jitter = jitter_img(images, repeat)
    dist_mat = dis_matrix_orbit(images_jitter)
    return dist_mat


def shortest_dist(dis_matrix):
    """ Find the shortest path between every pair
    :param dis_matrix: distance matrix based on data transformations
    :return: shortest distance matrix
    """
    dist_matrix_short = shortest_path(csgraph=dis_matrix, directed=False)
    print('Finish finding shortest path!')
    return dist_matrix_short


def cover_num(dis_matrix_short, epsilon, max_clusters):
    """ Calculate the sample covering number (more accurate version)
    :param dis_matrix_short: shortest distance matrix, N x N array
    :param epsilon: resolution
    :param max_clusters: maximum number of clusters to try
    :return: sample covering number, number of clusters
    """
    N = dis_matrix_short.shape[0]
    scn = N
    cnt = 0

    # Greedy search, gradually decrease the number of clusters, until the resulted covering number does not change
    for clusters_n in range(max_clusters, 0, -1):
        if clusters_n == 1:
            scn = 1
            break
        kmedoids = KMedoids(n_clusters=clusters_n, method='pam', init='k-medoids++', metric='precomputed').fit(
                 dis_matrix_short)
        centers = kmedoids.medoid_indices_
        labels = kmedoids.labels_
        out_cluster_idx = []
        for idx in centers:
            points = np.where(labels == labels[idx])[0]
            for p in points:
                if dis_matrix_short[idx, p] > epsilon:
                    out_cluster_idx.append(p)
        scn_new = len(out_cluster_idx) + clusters_n
        if scn_new < scn:
            scn = scn_new
            cnt = 0
        else:
            cnt += 1
            if cnt > 5:
                break
        # print(clusters_n, scn, scn_new)
    return scn


def cover_num_fast(dis_matrix_short, epsilon, clusters_n):
    """Calculate the sample covering number (faster version)
    :param dis_matrix_short: shortest distance matrix, N x N array
    :param epsilon: resolution
    :param clusters_n: number of clusters
    :return: sample covering number
    """
    clusters_n = int(clusters_n)
    # kmedoids = KMedoids(n_clusters=clusters_n, method='pam', init='k-medoids++', metric='precomputed').fit(dis_matrix_short)
    kmedoids = KMedoids(n_clusters=clusters_n, method='alternate', init='k-medoids++', metric='precomputed').fit(
        dis_matrix_short)
    centers = kmedoids.medoid_indices_
    labels = kmedoids.labels_
    out_cluster_idx = []
    for idx in centers:
        points = np.where(labels == labels[idx])[0]
        for p in points:
            if dis_matrix_short[idx, p] > epsilon:
                out_cluster_idx.append(p)
    scn = len(out_cluster_idx) + clusters_n
    # print(scn)

    return scn


def interclass_distance(dis_matrix, labels):
    """ Find the inter-class distance (smallest distance btw two classes) for each pair of classes
    :param dis_matrix: distance matrix, N x N array
    :param labels: N array
    :return: #class x #class array
    """
    classes = labels.max()
    dis_class = np.zeros((classes+1, classes+1))

    for i in range(classes):
        for j in range(i + 1, classes+1):
            indices_i = np.where(labels == i)[0]
            indices_j = np.where(labels == j)[0]
            dis = dis_matrix[indices_i[:, None], indices_j].min()
            dis_class[i, j] = dis_class[j, i] = dis

    return dis_class

