import h5py
import numpy as np
import tensorflow as tf


class Lab2DataLoader:
    train_path = 'lab2_train_data.h5'
    test_path = 'lab2_test_data.h5'
    keys = ['rgb', 'seg', 'color_codes']

    def loadData(self):
        train_file = h5py.File(self.train_path, 'r+')
        test_file = h5py.File(self.test_path, 'r+')
        assert train_file['rgb'].shape == (2975, 128, 256, 3)
        assert train_file['seg'].shape == (2975, 128, 256, 1)
        assert train_file['color_codes'].shape == (34, 3)
        assert test_file['rgb'].shape == (500, 128, 256, 3)
        assert test_file['seg'].shape == (500, 128, 256, 1)
        assert test_file['color_codes'].shape == (34, 3)
        train_img = np.array(train_file['rgb'])
        train_seg = np.array(train_file['seg'])
        test_img = np.array(test_file['rgb'])
        test_seg = np.array(test_file['seg'])
        cmap = np.array(train_file['color_codes'])
        train_file.close()
        test_file.close()
        return {'image': train_img, 'mask': train_seg}, {'image': test_img, 'mask': test_seg}, cmap
