from keras.engine import Model
import time

__author__ = 'xuwei'

import os
from common_convnet.utils.utils import console_logger
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from gender_classifier.imagenet_utils import preprocess_input

import tensorflow as tf

tf.python.control_flow_ops = tf

class ConvnetTrainBase(object):

    # convnet config
    _model_name = "Convnet_Base_Training_Class"
    _img_width = None
    _img_height = None
    _pretrained_model = None
    _complete_model = None


    # training config
    train_name = None

    train_data_dir = None
    validation_data_dir = None
    test_data_dir = None
    nb_train_samples = None
    nb_validation_samples =None
    nb_classes = None
    nb_epoch = None

    top_model_weights_path = None
    top_model_val_acc_checkpoint_path = None
    top_model_acc_checkpoint_path = None

    bottleneck_feature_train_path = None
    bottleneck_feature_validation_path = None

    val_acc_checkpoint_path = None
    acc_checkpoint_path = None
    final_model_path = None

    # assist
    _logger = console_logger()

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
                 nb_classes = None,
                 base_model_top_path=None,
                 base_model_complete_path=None
                 ):

        self.train_name = train_name

        logger = console_logger()
        logger.print_info("Initializing {0} named: {1}".format(self._model_name, str(self.train_name)))

        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.test_data_dir = test_data_dir
        self.nb_train_samples = nb_train_samples
        self.nb_validation_samples = nb_validation_samples
        self.nb_epoch = nb_epoch
        self.nb_classes = nb_classes

        self.top_model_val_acc_checkpoint_path = os.path.join(model_weight_folder, str(self._model_name)+"_"+str(self.train_name)+"_top_model_checkpoint_val_acc_{val_acc:.2f}.h5")
        self.top_model_acc_checkpoint_path = os.path.join(model_weight_folder, str(self._model_name)+"_"+str(self.train_name)+"_top_model_checkpoint_acc_{acc:.2f}.h5")
        self.top_model_weights_path = os.path.join(model_weight_folder, "{0}_{1}_top_model_weight.h5".format(self._model_name, self.train_name))

        self.val_acc_checkpoint_path = os.path.join(model_weight_folder, str(self._model_name)+"_"+str(self.train_name)+"_checkpoint_val_acc_{val_acc:.2f}.h5")
        self.acc_checkpoint_path = os.path.join(model_weight_folder, str(self._model_name)+"_"+str(self.train_name)+"_checkpoint_acc_{acc:.2f}.h5")
        self.final_model_path = os.path.join(model_weight_folder, "{0}_{1}_final_model.h5".format(self._model_name, self.train_name))

        if not bottleneck_feature_train_path:
            self.bottleneck_feature_train_path = os.path.join(model_weight_folder, "{0}_{1}_bottleneck_train.npy".format(self._model_name, self.train_name))
        else:
            self.bottleneck_feature_train_path = bottleneck_feature_train_path
        if not bottleneck_feature_validation_path:
            self.bottleneck_feature_validation_path = os.path.join(model_weight_folder, "{0}_{1}_bottleneck_validation.npy".format(self._model_name, self.train_name))
        else:
            self.bottleneck_feature_validation_path = bottleneck_feature_validation_path

        return


    def pretrain(self):

        assert self._pretrained_model!=None

        model = self._pretrained_model
        logger = console_logger()

        logger.print_info("start getting bottleneck features")

        logger.print_step("getting images from " + self.train_data_dir +" ...")
        datagen = ImageDataGenerator(rescale=1./255)
        raw_img_generator = datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self._img_width, self._img_height),
            batch_size=32,
            class_mode=None,   # classes are in sequence, first half 0, second half 1
            shuffle=False
        )

        logger.print_step("getting bottleneck features for training set ...")
        bottleneck_features_train = model.predict_generator(raw_img_generator, self.nb_train_samples)
        logger.print_step("saving bottleneck feature for training set to {} ...".format(self.bottleneck_feature_train_path))
        np.save(open(self.bottleneck_feature_train_path, 'w'), bottleneck_features_train)

        logger.print_step("getting images from " + self.validation_data_dir +" ...")
        generator = datagen.flow_from_directory(
                self.validation_data_dir,
                target_size=(self._img_width, self._img_height),
                batch_size=32,
                class_mode=None,
                shuffle=False)

        logger.print_step("getting bottleneck features for validation set ...")
        bottleneck_features_validation = model.predict_generator(generator, self.nb_validation_samples)
        logger.print_step("saving bottleneck validation feature for validation set to {} ...".format(self.bottleneck_feature_validation_path))
        np.save(open(self.bottleneck_feature_validation_path, 'w'), bottleneck_features_validation)

        return bottleneck_features_train.shape

    def get_bottleneck_feature_shape(self):
        train_data = np.load(open(self.bottleneck_feature_train_path))
        return train_data.shape[1:]

    def train_with_bottleneck_binary(self, model, optimizer, base_model_top_path=None):

        logger = console_logger()

        logger.print_step("loading bottleneck feature train numpy array from {}".format(self.bottleneck_feature_train_path))
        train_data = np.load(open(self.bottleneck_feature_train_path))
        train_labels = np.array([0] * (self.nb_train_samples / 2) + [1] * (self.nb_train_samples / 2))
        logger.print_step("loaded train data shape: " + str(train_data.shape))

        logger.print_step("loading bottleneck feature validation numpy array from {}".format(self.bottleneck_feature_validation_path))
        validation_data = np.load(open(self.bottleneck_feature_validation_path))
        validation_labels = np.array([0] * (self.nb_validation_samples / 2) + [1] * (self.nb_validation_samples / 2))

        logger.print_step("compile with optimizer {}".format(optimizer))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        # load validation accuracy checkpoint by default
        if base_model_top_path and os.path.exists(base_model_top_path):
            model.load_weights(base_model_top_path)
            logger.print_step("starting with existing weights from {} ...".format(base_model_top_path))

        callbacks_list = self._get_callback_list(top_layer_only=True)

        logger.print_step("training model ... (may take a while)")
        history = model.fit(
                train_data, train_labels,
                nb_epoch=self.nb_epoch,
                batch_size=32,
                validation_data=(validation_data, validation_labels),
                callbacks=callbacks_list)

        model.save_weights(self.top_model_weights_path)
        logger.print_info("top layer model saved to {}".format(self.top_model_weights_path))

        return model

    def train_with_bottleneck_multiclass(self, model, optimizer, base_model_top_path=None):

        assert self.nb_classes!=None, "Please provide the number of classes"

        def init_labels(sample_num, class_num):
            base = [0 for _ in range(class_num)]
            sample_per_class = sample_num//class_num
            assert sample_per_class*class_num==sample_num, "Sample not evenly distributed"
            result = []
            for i in range(class_num):
                label = base[:]
                label[i] = 1
                result += [label]*sample_per_class
            return np.array(result)


        logger = console_logger()

        logger.print_step("loading bottleneck feature train numpy array from {}".format(self.bottleneck_feature_train_path))
        train_data = np.load(open(self.bottleneck_feature_train_path))
        train_labels = init_labels(self.nb_train_samples, self.nb_classes)
        logger.print_step("loaded train data shape: " + str(train_data.shape))

        logger.print_step("loading bottleneck feature validation numpy array from {}".format(self.bottleneck_feature_validation_path))
        validation_data = np.load(open(self.bottleneck_feature_validation_path))
        validation_labels = init_labels(self.nb_validation_samples, self.nb_classes)

        logger.print_step("compile with optimizer {}".format(optimizer))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # load validation accuracy checkpoint by default
        if base_model_top_path and os.path.exists(base_model_top_path):
            model.load_weights(base_model_top_path)
            logger.print_step("starting with existing weights from {} ...".format(base_model_top_path))

        callbacks_list = self._get_callback_list(top_layer_only=True)

        logger.print_step("training model ... (may take a while)")
        history = model.fit(
                train_data, train_labels,
                nb_epoch=self.nb_epoch,
                batch_size=32,
                validation_data=(validation_data, validation_labels),
                callbacks=callbacks_list)

        model.save_weights(self.top_model_weights_path)
        logger.print_info("top layer model saved to {}".format(self.top_model_weights_path))

        return model

    def combine_top_layer_and_pretrained_model(self, top_model):
        pretrained = self._pretrained_model
        model = Model(input=pretrained.input, output=top_model(pretrained.outputs))
        model.save(self.final_model_path)
        self._logger.print_info("saved final model to {}".format(self.final_model_path))
        return model


    def train_with_frozen_binary(self, nb_frozen, base_model_path, optimizer):
        logger = console_logger()

        logger.print_info("start training full model with first {} frozen".format(nb_frozen))

        train_generator, validation_generator = self._get_train_validation_generator(
            batch_size=32,
            class_mode="binary"
        )


        model = load_model(base_model_path)
        for i in range(len(model.layers)):
            print str(i) + " : "+str(model.layers[i])

        for layer in model.layers[:nb_frozen]:
            layer.trainable = False

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks_list = self._get_callback_list(top_layer_only=False)

        logger.print_step("training model ... (may take a while)")
        history = model.fit_generator(
                    train_generator,
                    samples_per_epoch = self.nb_train_samples,
                    nb_epoch=self.nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=self.nb_validation_samples,
                    callbacks=callbacks_list
        )

        model.save(self.final_model_path)
        logger.print_info("full model saved to {}".format(self.final_model_path))

        return



    def train_with_frozen_multiclass(self, nb_frozen, base_model_path, optimizer):
        logger = console_logger()

        logger.print_info("start training full model with first {} frozen".format(nb_frozen))

        train_generator, validation_generator = self._get_train_validation_generator(
            batch_size=32,
            class_mode="categorical"
        )


        model = load_model(base_model_path)
        for i in range(len(model.layers)):
            print str(i) + " : "+str(model.layers[i])

        for layer in model.layers[:nb_frozen]:
            layer.trainable = False

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks_list = self._get_callback_list(top_layer_only=False)

        logger.print_step("training model ... (may take a while)")
        history = model.fit_generator(
                    train_generator,
                    samples_per_epoch = self.nb_train_samples,
                    nb_epoch=self.nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=self.nb_validation_samples,
                    callbacks=callbacks_list
        )

        model.save(self.final_model_path)
        logger.print_info("full model saved to {}".format(self.final_model_path))

        return



    def train_from_scratch_binary(self, optimizer):

        logger = console_logger()

        logger.print_info("start training from scratch")

        train_generator, validation_generator = self._get_train_validation_generator(
            batch_size=32,
            class_mode="binary"
        )

        assert self._complete_model != None, "the complete model should not be None"
        model = self._complete_model

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        callbacks_list = self._get_callback_list(top_layer_only=False)

        logger.print_step("training model ... (may take a while)")
        history = model.fit_generator(
                    train_generator,
                    samples_per_epoch = self.nb_train_samples,
                    nb_epoch=self.nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=self.nb_validation_samples,
                    callbacks=callbacks_list
        )

        model.save(self.final_model_path)
        logger.print_info("full model saved to {}".format(self.final_model_path))

        return

    def train_from_scratch_multiclass(self, optimizer):
        logger = console_logger()
        logger.print_info("start training from scratch")

        train_generator, validation_generator = self._get_train_validation_generator(
            batch_size=32,
            class_mode="categorical"
        )

        assert self._complete_model != None, "the complete model should not be None"
        model = self._complete_model

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        callbacks_list = self._get_callback_list(top_layer_only=False)

        logger.print_step("training model ... (may take a while)")
        history = model.fit_generator(
                    train_generator,
                    samples_per_epoch = self.nb_train_samples,
                    nb_epoch=self.nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=self.nb_validation_samples,
                    callbacks=callbacks_list
        )

        model.save(self.final_model_path)

        logger.print_info("full model saved to {}".format(self.final_model_path))

        return

    def predict_with_final_model(self, model_path):

        logger = console_logger()

        img_path_arr = []
        for root, dirs, files in os.walk(self.test_data_dir):
            for f in files:
                if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png"):
                    img_path = os.path.join(root, f)
                    img_path_arr.append(img_path)

        logger.print_info("{} test cases".format(len(img_path_arr)))

        logger.print_info("getting model from {}".format(model_path))
        model = load_model(model_path)

        test_result_arr = []

        logger.set_step(1)
        for img_path in img_path_arr:
            img = image.load_img(img_path, target_size=(self._img_width, self._img_height))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            x = x/255.                      # important - scale to 0-1

            start = time.time()
            preds = model.predict(x)
            time_lapse =time.time()-start

            test_result_arr.append((img_path, preds))
            logger.print_step("Preds: {} \n File: {} \n Time: {}".format(preds, img_path, time_lapse))

        return test_result_arr


    def set_pretrained_model(self, *args, **kwargs):
        pass

    def set_complete_model(self, *args, **kwargs):
        pass



    """
    private method
    """
    def _get_train_validation_generator(self, batch_size=32, class_mode="categorical"):
        logger = console_logger()

        logger.print_info("start constructing image generator, with batch size {0} and class mode {1}".format(batch_size, class_mode))

        datagen = ImageDataGenerator(rescale=1./255)

        logger.print_step("getting images from " + self.train_data_dir +" ...")
        train_generator = datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self._img_width, self._img_height),
            batch_size=batch_size,
            class_mode=class_mode
        )

        logger.print_step("getting images from " + self.train_data_dir +" ...")
        validation_generator = datagen.flow_from_directory(
                self.validation_data_dir,
                target_size=(self._img_width, self._img_height),
                batch_size=batch_size,
                class_mode=class_mode
        )

        return train_generator, validation_generator

    def _get_callback_list(self, top_layer_only=True):

        if top_layer_only:
            acc_chkpt_path = self.top_model_acc_checkpoint_path
            val_acc_chkpt_path = self.top_model_val_acc_checkpoint_path
        else:
            acc_chkpt_path = self.acc_checkpoint_path
            val_acc_chkpt_path = self.val_acc_checkpoint_path

        val_acc_checkpoint = ModelCheckpoint(val_acc_chkpt_path, monitor='val_acc', verbose=1, save_best_only=True,save_weights_only=False, mode='max')
        acc_checkpoint = ModelCheckpoint(acc_chkpt_path, monitor='acc', verbose=1, save_best_only=True,save_weights_only=False, mode='max')
        callbacks_list = [val_acc_checkpoint, acc_checkpoint]

        return callbacks_list