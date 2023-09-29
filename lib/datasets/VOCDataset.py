# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

from __future__ import print_function, division

import multiprocessing
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from datasets.transform import *


class VOCDataset(Dataset):
    def __init__(self, dataset_name, cfg, period, aug):
        self.dataset_name = dataset_name
        self.root_dir = os.path.join(cfg.ROOT_DIR, 'data', 'VOCdevkit')
        self.dataset_dir = os.path.join(self.root_dir, dataset_name)
        self.rst_dir = os.path.join(self.root_dir, 'results', dataset_name, 'Segmentation')
        self.eval_dir = os.path.join(self.root_dir, 'eval_result', dataset_name, 'Segmentation')
        self.period = period
        self.img_dir = os.path.join(self.dataset_dir, 'JPEGImages')
        self.ann_dir = os.path.join(self.dataset_dir, 'Annotations')
        self.seg_dir = os.path.join(self.dataset_dir, 'SegmentationClass')
        self.set_dir = os.path.join(self.dataset_dir, 'ImageSets', 'Segmentation')
        file_name = None
        if aug:
            file_name = self.set_dir + '/' + period + 'aug.txt'
        else:
            file_name = self.set_dir + '/' + period + '.txt'
        df = pd.read_csv(file_name, names=['filename'])
        self.name_list = df['filename'].values
        self.rescale = None
        self.centerlize = None
        self.randomcrop = None
        self.randomflip = None
        self.randomrotation = None
        self.randomscale = None
        self.randomhsv = None
        self.multiscale = None
        self.totensor = ToTensor()
        self.cfg = cfg

        if dataset_name == 'VOC2012':
            self.categories = [
                'aeroplane',  # 1
                'bicycle',  # 2
                'bird',  # 3
                'boat',  # 4
                'bottle',  # 5
                'bus',  # 6
                'car',  # 7
                'cat',  # 8
                'chair',  # 9
                'cow',  # 10
                'diningtable',  # 11
                'dog',  # 12
                'horse',  # 13
                'motorbike',  # 14
                'person',  # 15
                'pottedplant',  # 16
                'sheep',  # 17
                'sofa',  # 18
                'train',  # 19
                'tvmonitor']  # 20
            self.coco2voc = [[0],
                             [5],
                             [2],
                             [16],
                             [9],
                             [44],  # ,46,86],
                             [6],
                             [3],  # ,8],
                             [17],
                             [62],
                             [21],
                             [67],
                             [18],
                             [19],  # ,24],
                             [4],
                             [1],
                             [64],
                             [20],
                             [63],
                             [7],
                             [72]]

            self.num_categories = len(self.categories)
            assert (self.num_categories + 1 == self.cfg.MODEL_NUM_CLASSES)
            # TODO 2023/09/24 21:50
            self.cmap = self.__colormap(len(self.categories) + 1)

        if cfg.DATA_RESCALE > 0:
            self.rescale = Rescale(cfg.DATA_RESCALE, fix=False)
            # self.centerlize = Centerlize(cfg.DATA_RESCALE)
        if 'train' in self.period:
            if cfg.DATA_RANDOMCROP > 0:
                self.randomcrop = RandomCrop(cfg.DATA_RANDOMCROP)
            if cfg.DATA_RANDOMROTATION > 0:
                self.randomrotation = RandomRotation(cfg.DATA_RANDOMROTATION)
            if cfg.DATA_RANDOMSCALE != 1:
                self.randomscale = RandomScale(cfg.DATA_RANDOMSCALE)
            if cfg.DATA_RANDOMFLIP > 0:
                self.randomflip = RandomFlip(cfg.DATA_RANDOMFLIP)
            if cfg.DATA_RANDOM_H > 0 or cfg.DATA_RANDOM_S > 0 or cfg.DATA_RANDOM_V > 0:
                self.randomhsv = RandomHSV(cfg.DATA_RANDOM_H, cfg.DATA_RANDOM_S, cfg.DATA_RANDOM_V)
        else:
            self.multiscale = Multiscale(self.cfg.TEST_MULTISCALE)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        img_file = self.img_dir + '/' + name + '.jpg'
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = np.array(io.imread(img_file),dtype=np.uint8)
        r, c, _ = image.shape
        sample = {'image': image, 'name': name, 'row': r, 'col': c}

        if 'train' in self.period:
            seg_file = self.seg_dir + '/' + name + '.png'
            segmentation = np.array(Image.open(seg_file))
            sample['segmentation'] = segmentation

            if self.cfg.DATA_RANDOM_H > 0 or self.cfg.DATA_RANDOM_S > 0 or self.cfg.DATA_RANDOM_V > 0:
                sample = self.randomhsv(sample)
            if self.cfg.DATA_RANDOMFLIP > 0:
                sample = self.randomflip(sample)
            if self.cfg.DATA_RANDOMROTATION > 0:
                sample = self.randomrotation(sample)
            if self.cfg.DATA_RANDOMSCALE != 1:
                sample = self.randomscale(sample)
            if self.cfg.DATA_RANDOMCROP > 0:
                sample = self.randomcrop(sample)
            if self.cfg.DATA_RESCALE > 0:
                # sample = self.centerlize(sample)
                sample = self.rescale(sample)
        else:
            if self.cfg.DATA_RESCALE > 0:
                sample = self.rescale(sample)
            sample = self.multiscale(sample)

        if 'segmentation' in sample.keys():
            # 得到 boolean 掩码
            sample['mask'] = sample['segmentation'] < self.cfg.MODEL_NUM_CLASSES
            t = sample['segmentation']
            t[t >= self.cfg.MODEL_NUM_CLASSES] = 0
            sample['segmentation_onehot'] = onehot(t, self.cfg.MODEL_NUM_CLASSES)
        sample = self.totensor(sample)

        return sample

    def __colormap(self, N):
        """Get the map from label index to color

        Args:
            N: number of class

            return: a Nx3 matrix

        """
        cmap = np.zeros((N, 3), dtype=np.uint8)

        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        for i in range(N):
            r = 0
            g = 0
            b = 0
            idx = i
            for j in range(7):
                str_id = uint82bin(idx)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                idx = idx >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        return cmap

    # 将标签转换为颜色rgb三通道颜色
    def label2colormap(self, label):
        m = label.astype(np.uint8)
        r, c = m.shape  # row col
        cmap = np.zeros((r, c, 3), dtype=np.uint8)
        cmap[:, :, 0] = (m & 1) << 7 | (m & 8) << 3
        cmap[:, :, 1] = (m & 2) << 6 | (m & 16) << 2
        cmap[:, :, 2] = (m & 4) << 5
        return cmap

    def save_result(self, result_list, model_id):
        """Save test results

        Args:
            result_list(list of dict): [{'name':name1, 'predict':predict_seg1},{...},...]

        """
        i = 1
        folder_path = os.path.join(self.rst_dir, '%s_%s_cls' % (model_id, self.period))
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for sample in result_list:
            file_path = os.path.join(folder_path, '%s.png' % sample['name'])
            # predict_color = self.label2colormap(sample['predict'])
            # p = self.__coco2voc(sample['predict'])
            cv2.imwrite(file_path, sample['predict'])
            print('[%d/%d] %s saved' % (i, len(result_list), file_path))
            i += 1

    def do_matlab_eval(self, model_id):
        import subprocess
        path = os.path.join(self.root_dir, 'VOCcode')
        eval_filename = os.path.join(self.eval_dir, '%s_result.mat' % model_id)
        cmd = 'cd {} && '.format(path)
        cmd += 'matlab -nodisplay -nodesktop '
        cmd += '-r "dbstop if error; VOCinit; '
        cmd += 'VOCevalseg(VOCopts,\'{:s}\');'.format(model_id)
        cmd += 'accuracies,avacc,conf,rawcounts = VOCevalseg(VOCopts,\'{:s}\'); '.format(model_id)
        cmd += 'save(\'{:s}\',\'accuracies\',\'avacc\',\'conf\',\'rawcounts\'); '.format(eval_filename)
        cmd += 'quit;"'

        print('start subprocess for matlab evaluation...')
        print(cmd)
        subprocess.call(cmd, shell=True)

    def do_python_eval(self, model_id):
        predict_folder = os.path.join(self.rst_dir, '%s_%s_cls' % (model_id, self.period))
        gt_folder = self.seg_dir
        TP = []
        P = []
        T = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            # 多进程共享变量 加锁
            TP.append(multiprocessing.Value('i', 0, lock=True))
            P.append(multiprocessing.Value('i', 0, lock=True))
            T.append(multiprocessing.Value('i', 0, lock=True))

        def compare(start, step, TP, P, T):
            for idx in range(start, len(self.name_list), step):
                print('%d/%d' % (idx, len(self.name_list)))
                name = self.name_list[idx]
                predict_file = os.path.join(predict_folder, '%s.png' % name)
                gt_file = os.path.join(gt_folder, '%s.png' % name)
                predict = np.array(Image.open(predict_file))  # cv2.imread(predict_file)
                gt = np.array(Image.open(gt_file))
                # 变为 True False 的 mask
                # 使用 PIL 或者 imageio 读取图片会包含第四个通道 所以需要去除第四个通道
                cal = gt < 255
                mask = (predict == gt) * cal

                # 这里是彩色图像转换为单通道标签图像还是单通道标签图像转换为彩色图像？
                # 这里应该将彩色的真值图片转换为标签图片
                for i in range(self.cfg.MODEL_NUM_CLASSES):
                    P[i].acquire()
                    # * cls 把第四个通道去除
                    # 预测的第 i 个类别的像素个数
                    P[i].value += np.sum((predict == i) * cal)
                    P[i].release()
                    T[i].acquire()
                    # 第 i 个类别真正的像素个数
                    T[i].value += np.sum((gt == i) * cal)
                    T[i].release()
                    TP[i].acquire()
                    TP[i].value += np.sum((gt == i) * mask)
                    TP[i].release()

        p_list = []
        # 8个进程同时工作计算预测结果与真实结果
        for i in range(8):
            p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        IoU = []
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            # IoU 计算，真正预测成功的个数（即交集） 除以真值+预测个数-TP（即并集）
            IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
        for i in range(self.cfg.MODEL_NUM_CLASSES):
            if i == 0:
                print('%11s:%7.3f%%' % ('backbound', IoU[i] * 100), end='\t')
            else:
                if i % 2 != 1:
                    print('%11s:%7.3f%%' % (self.categories[i - 1], IoU[i] * 100), end='\t')
                else:
                    print('%11s:%7.3f%%' % (self.categories[i - 1], IoU[i] * 100))

        miou = np.mean(np.array(IoU))
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', miou * 100))

        # def do_python_eval(self, model_id):

    #    predict_folder = os.path.join(self.rst_dir,'%s_%s_cls'%(model_id,self.period))
    #    gt_folder = self.seg_dir
    #    TP = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
    #    P = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
    #    T = np.zeros((self.cfg.MODEL_NUM_CLASSES), np.uint64)
    #    for idx in range(len(self.name_list)):
    #        print('%d/%d'%(idx,len(self.name_list)))
    #        name = self.name_list[idx]
    #        predict_file = os.path.join(predict_folder,'%s.png'%name)
    #        gt_file = os.path.join(gt_folder,'%s.png'%name)
    #        predict = cv2.imread(predict_file)
    #        gt = cv2.imread(gt_file)
    #        cal = gt<255
    #        mask = (predict==gt) & cal
    #        for i in range(self.cfg.MODEL_NUM_CLASSES):
    #            P[i] += np.sum((predict==i)*cal)
    #            T[i] += np.sum((gt==i)*cal)
    #            TP[i] += np.sum((gt==i)*mask)
    #    TP = TP.astype(np.float64)
    #    T = T.astype(np.float64)
    #    P = P.astype(np.float64)
    #    IoU = TP/(T+P-TP)
    #    for i in range(self.cfg.MODEL_NUM_CLASSES):
    #        if i == 0:
    #            print('%15s:%7.3f%%'%('backbound',IoU[i]*100))
    #        else:
    #            print('%15s:%7.3f%%'%(self.categories[i-1],IoU[i]*100))
    #    miou = np.mean(IoU)
    #    print('==================================')
    #    print('%15s:%7.3f%%'%('mIoU',miou*100))

    def __coco2voc(self, m):
        r, c = m.shape
        result = np.zeros((r, c), dtype=np.uint8)
        for i in range(0, 21):
            for j in self.coco2voc[i]:
                result[m == j] = i
        return result
