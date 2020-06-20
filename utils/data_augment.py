import tensorflow as tf
import numpy as np

__all__ = ['normalize_image']

def normalize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image /= tf.constant(255., dtype=tf.float32)
    return image, label

def random_flip_left_right(image, label):
    return tf.image.random_flip_left_right(image), label

def random_flip_up_down(image, label):
    return tf.image.random_flip_up_down(image), label

def random_rotate_90(image, label):
    m = tf.random.uniform([], 0, 3, dtype=tf.int32)
    image = tf.image.rot90(image, k=m)
    return image, label

def gen_random_cutout(mask_size):
    def _random_cutout(image, label):
        mask_value = tf.reduce_mean(image)

        h, w, c = image.shape
        # top = tf.random.uniform([], 0-mask_size // 2, h-mask_size)
        # left = tf.random.uniform([], 0-mask_size // 2, w-mask_size)
        top = np.random.randint(0-mask_size // 2, h-mask_size)
        left = np.random.randint(0-mask_size // 2, w-mask_size)
        # bottom = top + mask_size
        # right = left + mask_size
        # top = tf.cast(top, tf.int32)
        # left = tf.cast(left, tf.int32)
        # bottom = tf.cast(bottom, tf.int32)
        # right = tf.cast(right, tf.int32)

        if top < 0:
            top = 0
        if left < 0:
            left = 0
        
        mask = tf.fill([mask_size, mask_size, c], mask_value)
        padding = tf.constant([[top, h-top-mask_size], [left, w-left-mask_size], [0, 0]])
        mask = tf.pad(mask, padding, 'CONSTANT')
        mask = tf.cast(mask, image.dtype)

        # image[top:bottom, left:right, :] = mask_value
        image = image - (image * mask)/mask_value + mask
        return image, label

    return _random_cutout


if __name__ == '__main__':
    import sys
    sys.path.append('./')

    import numpy as np
    import matplotlib.pyplot as plt
    from utils.datasets import load_data

    _random_cutout = gen_random_cutout(42)

    @tf.function
    def augment(image, label):
        # image, label = normalize_image(image, label)
        image, label = random_flip_left_right(image, label)
        image, label = random_flip_up_down(image, label)
        image, label =_random_cutout(image, label)
        image, label = random_rotate_90(image, label)
        return image, label

    x, y, _ = load_data('data/dataset/', classes=['normal', 'nude', 'swimwear'], size=128, cache_path='data/data_cache/dataset_size128.pkl', auto_pad_val=False)
    n_data = len(x)
    ds = tf.data.Dataset.from_tensor_slices((x, y)) \
        .shuffle(n_data).map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(64).prefetch(tf.data.experimental.AUTOTUNE).repeat(3)
    
    cnt = 0
    for inputs, labels in ds:
        cnt += 1
        if cnt > 5: break
        inputs = inputs.numpy()
        plt.imshow(np.uint8(inputs[0]*255))
        plt.savefig('{}.png'.format(cnt))
 
