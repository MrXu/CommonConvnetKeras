from __future__ import print_function
import os

__author__ = 'xuwei'

from keras.utils.visualize_util import plot


def plot_model_cwd(model, dir_path, image_name):
    plot(model, to_file=os.path.join(dir_path, image_name), show_shapes=True, show_layer_names=True)


class console_logger(object):
    step = 1

    def __init__(self, step=1):
        self.step = step

    def set_step(self, step):
        self.step = step

    def print_step(self, message):
        print(">step {} : {}".format(str(self.step), str(message)))
        self.step+=1

    def print_info(self, msg):
        print("[info] {}".format(str(msg)))

    def print_error(self, msg):
        print("[error] {}".format(str(msg)))