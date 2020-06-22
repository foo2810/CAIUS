import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


def plot_learning_curve(filename, epochs, stack=None, history=None, dark=False):
    if stack is None and history is None:
        return

    if stack is not None:
        history = stack.history

    if dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    e = range(epochs)
    plt.figure(figsize=(10, 12))
    plt.subplot(2, 1, 1)
    sns.lineplot(e, history['train_acc'], label="accuracy", color='darkcyan')
    sns.lineplot(e, history['test_acc'], label='val_accuracy', color='coral')
    plt.title("accuracy")
    plt.xlabel("epoch")
    plt.legend(loc='best')

    plt.subplot(2, 1, 2)
    sns.lineplot(e, history['train_loss'], label="loss", color='darkcyan')
    sns.lineplot(e, history['test_loss'], label='val_loss', color='coral')
    plt.title("loss")
    plt.xlabel("epoch")
    plt.legend(loc='best')

    plt.savefig(filename)
    plt.show()
    plt.figure()
