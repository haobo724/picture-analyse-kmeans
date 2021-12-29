import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans
from matplotlib import pyplot as plt
from hist import centroid_histogram
from draw import plot_colors
import numpy as np


def load_image(name, width=300):
    im = cv2.imread(name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_origin = im.copy()
    ratio = im.shape[0] / im.shape[1]
    if im.shape[0] * im.shape[1] > 640 * 960:
        height = int(width * ratio)
        im = cv2.resize(im, (width, height))
    return im, im_origin


def flatten(im):
    im_flatten = im.reshape(im.shape[0] * im.shape[1], 3)
    return im_flatten


def do_cluster(im_flatten, num_bins):
    # cluster = KMeans(n_clusters=num_bins)
    cluster = MiniBatchKMeans(n_clusters=num_bins, batch_size=2048)
    cluster.fit(im_flatten)
    return cluster


def visualisation(pic):
    plt.figure()
    plt.axis("off")
    plt.imshow(pic)
    plt.show()


if __name__ == '__main__':
    img_path = r'F:\FOTO\0206\P3310746-2.jpg'

    im, im_ori = load_image(img_path)
    color_nr = 6
    im_flatten = flatten(im)
    cluster = do_cluster(im_flatten, color_nr)
    hist = centroid_histogram(cluster)

    origin_width = im_ori.shape[1]
    bar = plot_colors(hist, cluster.cluster_centers_, percent_flag=False, width=origin_width)

    ratio = bar.shape[0] / bar.shape[1]
    height = int(origin_width * ratio)
    bar = cv2.resize(bar, (origin_width, height))
    white_block = np.ones_like(bar) * 255
    temp = np.concatenate((im_ori, white_block))
    temp = np.concatenate((temp, bar))

    visualisation(temp)
