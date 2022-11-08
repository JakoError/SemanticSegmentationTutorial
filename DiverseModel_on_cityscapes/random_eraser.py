import numpy as np


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img = input_img.copy()
        if img.ndim == 3:
            img_h, img_w, img_c = img.shape
        elif img.ndim == 2:
            img_h, img_w = img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        c=0

        img[top:top + h, left:left + w] = c

        return img

    return eraser