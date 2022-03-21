import os
import matplotlib.pyplot as plt


def crt_folder(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    pass


def graph(history, log_dir, fig_name):
    loss_train = history['loss']
    loss_val = history['val_loss']
    epoc = range(1, len(loss_train) + 1)
    plt.plot(epoc, loss_train, 'g', label='Training loss')
    plt.plot(epoc, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig = plt.gcf()
    fig.savefig("experiments/" + log_dir + fig_name)
    plt.clf()
