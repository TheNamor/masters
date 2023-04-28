#!/usr/bin/env python
# coding: utf-8

import numpy as np
from itertools import repeat
from matplotlib import pyplot as plt
from skimage.io import imread
import networkx as nx
import os
from tqdm import tqdm
from skimage.transform import resize

from Melanoma_cellgraph_globalfeats_functions import getGraphFeats

def plot_graph(points, edges, image=None):
    if image is not None:
        plt.imshow(image)
    for node1, node2 in edges:
        x1, y1 = points[node1]
        x2, y2 = points[node2]
        plt.plot([x1, x2], [y1, y2], "k.-", ms=3)
    plt.gca().set_aspect('equal')
    plt.show()

def get_edge_list(points, distance):
    edge_list_1st_row = []
    edge_list_2nd_row = []
    dist_list = []
    for i in range(len(points)):
        dist = np.sqrt((points[i,0] - points[:,0])**2 + (points[i,1] - points[:,1])**2)
        dist = dist.reshape(-1,1)
        #print(dist)
        x = np.where(dist<=distance)
        node_index = list(x[0])
        dist_list.append(np.average(dist[node_index]))
        edge_list_2nd_row.append(node_index)
        edge_list_1st_row.extend(repeat(i,len(node_index)))
    edge_list_2nd_row = [item for sublist in edge_list_2nd_row for item in sublist]
    edge_list = [edge_list_1st_row, edge_list_2nd_row]
    edge_list = np.array(edge_list)
    dist_list = np.array(dist_list)
    dist_list = dist_list.reshape(-1,1)
    return edge_list, dist_list

def full_edge_list(C):
    distance = 64.0
    edges, _ = get_edge_list(C, distance)
    full_edge = np.array(edges).T
    return full_edge

def get_graph(edge_list):
    G = nx.Graph()
    G.add_edges_from(edge_list)
    return G

def get_pointlists():
    order = list(range(24)) + [25, 26, 27, 30, 24, 28, 29] + list(range(31, 150))
    with open("C:/Users/TheNa/Desktop/properties.csv", "r") as infile:
        image_num = 0
        next(infile)
        line = infile.readline().strip().split(",")
        while len(line):
            points = []
            while line[0] == str(order[image_num]):
                points.append([float(line[2]), float(line[3])])
                line = infile.readline().split(",")
            yield points
            image_num += 1


IMG_WIDTH = 7680
IMG_HEIGHT = 4608
IMG_CHANNELS = 3

CRC_PATH = 'C:/Users/TheNa/Desktop/CRC_Dataset'
crc_ids = os.listdir(CRC_PATH)
DISPLAY_IMGS = False
START_AT = 107

crc_images = []

if DISPLAY_IMGS:
    print("\nReading images...")
    for n, id_ in tqdm(enumerate(crc_ids), total=len(crc_ids)):
        img = imread(CRC_PATH + "/" + id_)[:,:,:IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        crc_images.append(img)


points_generator = get_pointlists()

for i in range(START_AT):
    next(points_generator)

with open("C:/Users/TheNa/Desktop/features.csv", "a") as outfile:

    for i in tqdm(range(len(crc_ids)-START_AT)):
        print(f"\nReading points for image {START_AT+i}...")
        points = np.array(next(points_generator))
        print(f"\tRead in {len(points)} points!")
        if len(points) < 100:
            print("Point reading error:", points)
            break
        print("Finding edges...")
        edges = full_edge_list(points)
        print("Generating graph...")
        graph = get_graph(edges)
        print("Calculating features...")
        f = getGraphFeats(graph, library=0, bool_spect=True)
        f = f.reshape(1,-1)[0]
        print("Saving...")
        out = ""
        for feat in f:
            out += str(feat) + ","
        out = out[:-1] + "\n"
        outfile.write(out)
        outfile.flush()

plt.plot([1], [1])
plt.show()
