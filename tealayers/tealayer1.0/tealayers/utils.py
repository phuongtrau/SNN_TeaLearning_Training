import numpy as np
from numpy import newaxis
import random
import os
import PIL
from PIL import ImageOps, Image
import matplotlib.pyplot as plt
from scipy import ndimage

from torchvision.transforms import ToPILImage

position_i = ["justAPlaceholder", "symbol_1", "symbol_2",
              "symbol_3", "symbol_4", "symbol_5",
              "symbol_6", "symbol_7", "symbol_8",
              "symbol_9", "symbol_10", "symbol_11",
              "symbol_12", "symbol_13", "symbol_14",
              "symbol_15", "symbol_16", "symbol_17"]


def token_position(x):

    return int(x.split('_')[-1]) - 1


def subject_encode(x):

    return int(x[1:]) - 1


def export_data(data_dir='/content/drive/MyDrive/pose_classification/data/a-pressure-map-dataset-for-in-bed-posture-classification-1.0.0/experiment-i',
                preprocess=True):
    data_dict = dict()
    for _, dirs, _ in os.walk(data_dir):
        for directory in dirs:
            # each directory is a subject
            subject = directory
            data = None
            labels = None
            for _, _, files in os.walk(os.path.join(data_dir, directory)):
                for file in files:
                    file_path = os.path.join(data_dir, directory, file)
                    with open(file_path, 'r') as f:
                        lines = f.read().splitlines()[2:]
                        for i in range(3, len(lines) - 3):
                            
                            raw_data = np.fromstring(lines[i], dtype=float, sep='\t').reshape(64, 32)
                            
                            if preprocess is True:
                                past_image = np.fromstring(lines[i-1], dtype=float, sep='\t').reshape(64, 32)
                                future_image = np.fromstring(lines[i+1], dtype=float, sep='\t').reshape(64, 32)
                                
                                # Spatio-temporal median filter 3x3x3
                                raw_data = ndimage.median_filter(raw_data, 3)
                                past_image = ndimage.median_filter(past_image, 3)
                                future_image = ndimage.median_filter(future_image, 3)
                                raw_data = np.concatenate((raw_data[newaxis, :, :], past_image[newaxis, :, :], future_image[newaxis, :, :]), axis=0)
                                raw_data = np.median(raw_data, axis=0)

                            # Change the range from [0-1000] to [0-255].
                            file_data = np.round(raw_data * 255 / 1000).astype(np.uint8)
                            file_data = file_data.reshape(1, 64, 32)

                            # Turn the file index into position list,
                            # and turn position list into reduced indices.
                            file_label = token_position(position_i[int(file[:-4])])
                            file_label = np.array([file_label])

                            if data is None:
                                data = file_data
                            else:
                                data = np.concatenate((data, file_data), axis=0)
                            if labels is None:
                                labels = file_label
                            else:
                                labels = np.concatenate((labels, file_label), axis=0)

            data_dict[subject] = (data, labels)

    return data_dict


def flip(image):

    return ImageOps.flip(image)


def translate_x(image, p=0.5):
    value = image.size[0]
    if random.random() <= p:
        value = -value

    return image.transform(image.size, PIL.Image.AFFINE, (1, 0, value, 0, 1, 0))


def translate_y(image, p=0.5):
    value = image.size[1] * 0.1
    if random.random() <= p:
        value = -value

    return image.transform(image.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, value))


def rotation(image, p=0.5):
    value = 25
    if random.random() <= p:
        value = -value

    return image.rotate(value)


def cutout(img, n_holes, length):
        h = img.shape[1]
        w = img.shape[2]

        mask = np.ones((h, w), np.float32)

        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        
        return img

if __name__ == '__main__':
    data = export_data()
    print(data.keys())
    pass
