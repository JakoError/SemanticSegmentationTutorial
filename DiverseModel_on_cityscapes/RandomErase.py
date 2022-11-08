import tensorflow as tf
from keras import backend
from keras.engine import base_layer
from keras.engine import base_preprocessing_layer
from keras.layers.preprocessing import preprocessing_utils as utils

import numpy as np


class RandomErase(base_layer.BaseRandomLayer):
    def __init__(self, p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, seed=None,
                 **kwargs):
        base_preprocessing_layer.keras_kpl_gauge.get_cell('RandomErase').set(True)
        super(RandomErase, self).__init__(**kwargs, autocast=False, seed=seed,
                                          force_generator=True)
        self.p = p
        self.s_l = s_l
        self.s_h = s_h
        self.r_1 = r_1
        self.r_2 = r_2
        self.v_l = v_l
        self.v_h = v_h
        self.seed = seed

    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()
        inputs = utils.ensure_tensor(inputs, dtype=self.compute_dtype)
        input_shape = tf.shape(inputs)

        def random_erase():
            dtype = input_shape.dtype
            rands = self._random_generator.random_uniform([2], 0, dtype.max, dtype)

            img = input.copy()
            if img.ndim == 3:
                img_h, img_w, img_c = img.shape
            elif img.ndim == 2:
                img_h, img_w = img.shape

            p_1 = self._random_generator.random_uniform(0, 1)

            if p_1 > self.p:
                return img

            while True:
                s = self._random_generator.random_uniform(self.s_l, self.s_h) * self.img_h * self.img_w
                r = self._random_generator.random_uniform(self.r_1, self.r_2)
                w = int(np.sqrt(s / r))
                h = int(np.sqrt(s * r))
                left = np.random.randint(0, self.img_w)
                top = np.random.randint(0, self.img_h)

                if left + w <= self.img_w and top + h <= self.img_h:
                    break

            c = 0

            img[top:top + h, left:left + w] = c

            return img
