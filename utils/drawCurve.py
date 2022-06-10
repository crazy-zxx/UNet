import os

import matplotlib.pyplot as plt
import numpy as np


def draw(epochs, value_list, label_list, title, x_label, y_label, color_list, save_path):
    fig = plt.figure()
    x = range(1, epochs + 1)
    for y, l, c in zip(value_list, label_list, color_list):
        plt.plot(x, y, color=c, label=l)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc="best")  # place the legend in the right place
    plt.xlim(0.5, epochs + 0.5)
    # plt.ylim(0, 1)
    # x轴间隔1显示
    # x_major_locator = MultipleLocator(1)
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(x_major_locator)

    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{title}.png'))
    # plt.show()
    plt.close(fig)


if __name__ == '__main__':
    draw(10, [np.random.rand(10), np.random.rand(10), np.random.rand(10)], ['a', 'b', 'c'],
         'test_all', 'epoch', 'acc', ['red', 'blue', 'green'], '.')
