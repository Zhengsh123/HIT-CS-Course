#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/6/1 22:20
# @Author  : ZSH
'''
读取xml原始数据，生成更适合使用的txt格式文件
'''
import xml.etree.ElementTree as ElementTree
import os

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')


def parse_rec(file_path):
    """
    读取VOC数据集每个数据集的信息
    :param file_path: VOC数据集索引文件的地址，如'./data/train/Annotations/2007_000629.xml'
    :return:
    """
    tree = ElementTree.parse(file_path)
    objects = []
    filename = tree.find('filename').text
    for obj in tree.findall('object'):
        obj_struct = dict()
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['box'] = [int(float(bbox.find('xmin').text)), int(float(bbox.find('ymin').text)),
                             int(float(bbox.find('xmax').text)), int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)
    return objects, str(filename)


def read_xml(annotation_path, txt_path):
    """
    读取xml文件中需要使用的信息，使用txt文件形式保存
    :param annotation_path: annotation文件夹地址
    :param txt_path: txt文件想保存的地址
    :return:
    """
    file_list=[os.path.join(annotation_path, file) for file in os.listdir(annotation_path)]
    for file in file_list:
        objects, filename = parse_rec(file)
        with open(txt_path, 'a+')as fw:
            fw.write(filename + ' ')
            for obj in objects:
                class_name = obj['name']
                box = obj['box']
                class_name = CLASSES.index(class_name)
                fw.write(' ' + str(box[0]) + ' ' + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' ' + str(
                    class_name))
            fw.write('\n')
        fw.close()


if __name__ == "__main__":
    read_xml('./data/train/Annotations','./train.txt')
    read_xml('./data/test/Annotations', './test.txt')
