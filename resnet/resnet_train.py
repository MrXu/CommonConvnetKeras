from common_convnet.resnet.resnet import ResNetBuilder

__author__ = 'xuwei'

from common_convnet.ConvnetTrainBase import ConvnetTrainBase

class Resnet34Train(ConvnetTrainBase):
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
        self._model_name = "resnet34"
        self._img_height = 224
        self._img_width = 224

        # training config
        super(Resnet34Train, self).__init__(train_name, train_data_dir, validation_data_dir, test_data_dir,
                                                nb_train_samples, nb_validation_samples, nb_epoch, model_weight_folder, bottleneck_feature_train_path, bottleneck_feature_validation_path)


    def set_complete_model(self, output_number):
        if output_number==1:
            self._complete_model = ResNetBuilder.build_resnet_34_binary(
                input_shape = (self._img_width, self._img_height, 3)
            )
        else:
            self._complete_model = ResNetBuilder.build_resnet_34(
                input_shape = (self._img_width, self._img_height, 3),
                num_outputs=output_number
            )


class Resnet50Train(ConvnetTrainBase):
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
        self._model_name = "resnet50"
        self._img_height = 224
        self._img_width = 224

        # training config
        super(Resnet50Train, self).__init__(train_name, train_data_dir, validation_data_dir, test_data_dir,
                                                nb_train_samples, nb_validation_samples, nb_epoch, model_weight_folder, bottleneck_feature_train_path, bottleneck_feature_validation_path)


    def set_complete_model(self, output_number):
        self._complete_model = ResNetBuilder.build_resnet_50(
            (3, self._img_width, self._img_height),
            output_number
        )