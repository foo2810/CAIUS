import sys
import cv2
import numpy as np
import pickle
from pathlib import Path

__all__ = ['load_data', 'train_test_split']

def resize_img(cv_img, size=512):
    h, w = cv_img.shape[:2]
    max_edge = max(h, w)

    if max_edge < size:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA

    if h > w:
        resized_shape = (int(w/h*size), size)
    else:
        resized_shape = (size, int(h/w*size))

    resized_img = cv2.resize(cv_img, resized_shape, interpolation=interpolation)

    return resized_img

def padding(cv_img, size=512, auto_pad_val=False):
    h, w = cv_img.shape[:2]
    if cv_img.ndim == 3:
        base_shape = (size, size, cv_img.shape[2])
    elif cv_img.ndim == 2:
        base_shape = (size, size)
    
    pad_val = 0
    if auto_pad_val: pad_val = np.mean(cv_img)
    padded = np.zeros(base_shape, dtype=np.uint8) + pad_val
    if h > w:
        pad_edge = (size - w) // 2
        idx = slice(pad_edge, pad_edge+w)
        padded[:, idx, :] += cv_img - pad_val
    else:
        pad_edge = (size - h) // 2
        idx = slice(pad_edge, pad_edge+h)
        padded[idx, :, :] += cv_img - pad_val

    return padded

def _load_raw_data(directory, classes, size, auto_pad_val):
    data = dict(zip(list(range(len(classes))), [[] for _ in range(len(classes))]))
    id2class = dict(zip(list(range(len(classes))), classes))
    class2id = dict(zip(classes, list(range(len(classes)))))

    for cls_name in classes:
        class_path = directory / cls_name / 'org'

        for cnt, img_path in enumerate(class_path.glob('*')):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                sys.stderr.write('Cannot read file correctly. : {}\n'.format(img_path))
                continue
            resized = resize_img(img, size=size)
            padded = padding(resized, size=size, auto_pad_val=auto_pad_val)
            padded = cv2.cvtColor(np.uint8(padded), cv2.COLOR_BGR2RGB).astype(np.float32)

            data[class2id[cls_name]] += [padded]  # (W, H, C)
    return data, id2class

def load_data(directory, classes=('Normal', 'Nude', 'Swimwear'), size=512, cache_path=None, auto_pad_val=False):
    if type(directory) is str:
        directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(str(directory))

    classes = tuple(map(lambda x: str.upper(x), classes))
    if not (set(classes) < set(['NORMAL', 'NUDE', 'SWIMWEAR', 'MINOR'])):
        raise ValueError('Included unknown class names')

    if cache_path is not None:
        if type(cache_path) is str: cache_path = Path(cache_path)
        if not cache_path.exists():
            data, id2class = _load_raw_data(directory, classes, size, auto_pad_val)

            with cache_path.open('wb') as fp:
                pickle.dump((data, id2class), fp, protocol=4)
        else:
            with cache_path.open('rb') as fp:
                data, id2class = pickle.load(fp)
    else:
        data, id2class = _load_raw_data(directory, classes, size, auto_pad_val)
    
    x, y = [], []
    for class_id in data.keys():
        x += data[class_id]
        y += [class_id for _ in range(len(data[class_id]))]
    
    x, y = np.array(x, dtype=np.float), np.array(y, dtype=np.int)
    
    return x, y, id2class

def train_test_split(x, y, train_rate):
    p = np.random.permutation(len(x))
    x, y = x[p], y[p]
    n_train = int(len(x) * train_rate)
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    x, y, _ = load_data('data/dataset/', classes=['normal', 'nude', 'swimwear'], size=512, cache_path='data/data_cache/dataset_size256.pkl')
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_rate=0.5)

    print('x: ', x.shape)
    print('y: ', y.shape)

    print('x_train: ', x_train.shape)
    print('x_test: ', x_test.shape)

    print(y_train)
