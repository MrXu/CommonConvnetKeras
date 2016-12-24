__author__ = 'xuwei'

from common_convnet.ConvnetTrainBase import ConvnetTrainBase

from common_convnet.squeezenet.squeezenet_v1 import get_squeezenet_pretrained
import tensorflow as tf

tf.python.control_flow_ops = tf

class SqueezenetTrainV1(ConvnetTrainBase):


    def __init__(self,
                 train_name,
                 train_data_dir,
                 validation_data_dir,
                 test_data_dir,
                 nb_train_samples,
                 nb_validation_samples,
                 nb_epoch,
                 model_weight_folder,
                 bottleneck_feature_train_path=None,
                 bottleneck_feature_validation_path=None,
                 nb_classes = None
                 ):

        # model config
        self._model_name = "squeezenet_v1"
        self._img_height = 227
        self._img_width = 227
        self._pretrained_model = get_squeezenet_pretrained()

        # training config
        super(SqueezenetTrainV1, self).__init__(
            train_name,
            train_data_dir,
            validation_data_dir,
            test_data_dir,
            nb_train_samples,
            nb_validation_samples,
            nb_epoch,
            model_weight_folder,
            bottleneck_feature_train_path,
            bottleneck_feature_validation_path,
            nb_classes)
