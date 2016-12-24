## CommonConvnetKeras
This repository intends to implement common convolutional neural networks for image classification.

### Dependency
1. Tensorflow 
2. Keras 
3. Opencv

### Structure
1. ConvnetTrainBase is the base class for all convnet implementation. 
2. Each folder contains the implementation of convnet and their training configuration

### Usage
1. training:
```
train_data_dir = "/path/to/training/dataset"
validation_data_dir = "/path/to/validation/dataset"
test_data_dir = "/path/to/test/dataset"
nb_train_samples = 2000
nb_validation_samples = 600

dir_path = os.path.dirname(os.path.realpath(__file__))


def experiment_1():

    resnet = Resnet34Train(
        train_name = "exp_1",
        train_data_dir = train_data_dir,
        validation_data_dir = validation_data_dir,
        test_data_dir = test_data_dir,
        nb_train_samples = nb_train_samples,
        nb_validation_samples = nb_validation_samples,
        nb_epoch = 100
    )

    # binary
    resnet.set_complete_model(1)

    optimizer = "rmsprop"
    resnet.train_from_scratch_binary(
        optimizer=optimizer
    )

    return
```

2. Prediction
```
def experiment_1():

    densenet_fast = DensenetFastTrain(
        train_name = "exp_1",
        train_data_dir = train_data_dir,
        validation_data_dir = validation_data_dir,
        test_data_dir = test_data_dir,
        nb_train_samples = nb_train_samples,
        nb_validation_samples = nb_validation_samples,
        nb_epoch = 100,
        model_weight_folder = dir_path,
    )

    # predict
    final_model = os.path.join(dir_path, "densenet_fast_exp_1.h5")
    densenet_fast.predict_with_final_model(final_model)
```

### TODO
1. training flow
