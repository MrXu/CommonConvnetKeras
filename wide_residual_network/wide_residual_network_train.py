from common_convnet.ConvnetTrainBase import ConvnetTrainBase
from common_convnet.wide_residual_network.wide_residual_network import get_wrn_16_8_complete

__author__ = 'xuwei'



class WideResnet_16_8(ConvnetTrainBase):
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
                 bottleneck_feature_validation_path=None
                 ):

        # model config
        self._model_name = "wrn_16_8"
        self._img_height = 224
        self._img_width = 224

        # training config
        super(WideResnet_16_8, self).__init__(train_name, train_data_dir, validation_data_dir, test_data_dir,
                                                nb_train_samples, nb_validation_samples, nb_epoch, model_weight_folder, bottleneck_feature_train_path, bottleneck_feature_validation_path)


    def set_complete_model(self, output_number):
        self._complete_model = get_wrn_16_8_complete(
            nb_class = output_number,
            input_dim = (self._img_width, self._img_height, 3)
        )


