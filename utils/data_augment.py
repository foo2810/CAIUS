import tensorflow as tf

__all__ = ['normallize_image']

def normalize_image(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image /= tf.constant(255., dtype=tf.float32)
    return image, label

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.datasets import load_data

    def augment(image, label):
        image, label = normalize_image(image, label)
        return image, label

    x, y, _ = load_data('data/dataset/', classes=['normal', 'nude', 'swimwear'], size=128, cache_path='data/data_cache/dataset_size128_autopad.pkl', auto_pad_val=True)
    n_data = len(x)
    ds = tf.data.Dataset.from_tensor_slices((x, y)) \
        .shuffle(n_data).map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(100).prefetch(tf.data.experimental.AUTOTUNE)

    for inputs, labels in ds:
        inputs = inputs.numpy()
        print(inputs[0])
        plt.imshow(np.uint8(inputs[0]*255))
        break
 
