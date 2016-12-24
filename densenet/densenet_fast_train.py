from common_convnet.ConvnetTrainBase import ConvnetTrainBase
from common_convnet.densenet.densenet_fast import create_densenet_fast

__author__ = 'xuwei'



class DensenetFastTrain(ConvnetTrainBase):
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
        self._model_name = "densenet_fast"
        self._img_height = 224
        self._img_width = 224

        # training config
        super(DensenetFastTrain, self).__init__(train_name, train_data_dir, validation_data_dir, test_data_dir,
                                                nb_train_samples, nb_validation_samples, nb_epoch, model_weight_folder, bottleneck_feature_train_path, bottleneck_feature_validation_path)


    def set_complete_model(self, output_number):
        self._complete_model = create_densenet_fast(
            nb_classes=output_number,
            img_dim=(self._img_width, self._img_height, 3),
            depth=10
        )


