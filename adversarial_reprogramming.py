"""Finds adversarial programs for random networks;
repurposing the networks to work on different datasets such as mnist.

Can be called with the following arguments:
    --network (str): The network architecture to use.
    --dataset (str): The dataset to use. Defaults to mnist.
    --input_value_range (float): A value between 0 and 1 indicating how the pixel
                                 values of the input are scaled.
                                 If 0, the input is ignored and only the adversarial
                                 program is fed into core_model.
                                 If 1, in those pixels that contain the input image,
                                 there is no adversarial program used.
                                 Defaults to 1.
    --image_size (float): The size of the image relative to the input size expected
                          by the network. Defaults to 1.
    --epochs (int): The number of epochs to use to find an adversarial program.
                    Defaults to 20.
    --lr (float): The learning rate to use for training. Defaults to 0.01.
    --batch_size (int): The batch size to use. Defaults to 50.

    for example:
        python adversarial_reprogramming.py --network ResNet50V2 --dataset mnist \
            --input_value_range 0.05 --epochs 5

    Used with:
        tensorflow==2.8.1
        tensorflow-datasets==4.5.2
        tensorflow-probability==0.16.0
"""

import argparse
import math
from functools import partial

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp


def find_input_shape(model):
    """Return the input shape that the model expects."""
    config = model.get_config()
    return config["layers"][0]["config"]["batch_input_shape"][1:]


def condition_batchnormalization_layers(model, batch_size=50):
    """Takes a model, sets the momentum of all batch normalization layers to 0,
    and feeds a single batch of random data throughout the model to cause the
    batch normalization layers to update their moving mean and variance values.
    """

    def random_image_generator(shape=(224, 224, 3)):
        while True:
            yield tf.convert_to_tensor(np.random.randint(256, size=shape)), tf.one_hot(
                np.random.randint(1000), 1000
            )

    # set the momentum of each batch normalization layer to 0
    # such that the stored moving average of the mean and standard deviation simply
    # becomes the mean and standard deviation of the last batch seen during training.
    for layer in [
        layer
        for layer in model.layers[-1].layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    ]:
        layer.momentum = 0.0

    # set learning rate to 0 so that no weights are updated
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.0, momentum=0.0),
        loss=tf.keras.losses.CategoricalCrossentropy(),
    )

    input_shape = find_input_shape(model)

    dataset = tf.data.Dataset.from_generator(
        partial(random_image_generator, shape=input_shape),
        output_signature=(
            tf.TensorSpec(shape=input_shape, dtype=tf.int32),
            tf.TensorSpec(shape=(1000,), dtype=tf.int32),
        ),
    )
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    dataset = dataset.batch(batch_size).with_options(options)

    model.fit(dataset, epochs=1, steps_per_epoch=1, verbose=2)

    return model


# pylint: disable=too-few-public-methods
class InverseSoftSign(tf.keras.initializers.Initializer):
    """Initializer that generates a tensor
    with values drawn from the inverse softsign function."""

    def __call__(self, shape, dtype=None, **kwargs):
        # define distribution. Could possibly be moved to constructor,
        # but should not matter much for our use case.
        inversesoftsign = tfp.distributions.TransformedDistribution(
            distribution=tfp.distributions.Uniform(low=-1, high=1),
            bijector=tfp.bijectors.Invert(tfp.bijectors.Softsign()),
        )
        return inversesoftsign.sample(shape)


class AdditiveProgram(tf.keras.layers.Layer):
    """A layer that augments the input with a (trainable) adversarial program"""

    def __init__(self, input_value_range=1.0, model=None, image_mask=None, **kwargs):
        super().__init__(**kwargs)
        self.input_value_range = input_value_range
        self.model = model
        self.image_mask = image_mask
        self.program = None

    def build(self, input_shape):
        """Create trainable weights for the program."""

        self.program = self.add_weight(
            name="program",
            shape=input_shape[1:],
            initializer=InverseSoftSign(),
            trainable=True,
        )

    @tf.function
    def get_program(self):
        """Returns the adversarial program with pixel values between 0 and 1."""

        return (tf.nn.softsign(self.program) + 1) / 2

    @tf.function
    def combine_program_with_image(self, image):
        """convex combination of input and adversarial program"""

        return (
            1 - self.input_value_range * self.image_mask
        ) * self.get_program() * 255 + self.input_value_range * image

    @tf.function
    def call(self, inputs):
        """Combine the input with the program and feed through the core model
        if it exists."""

        inputs_combined_with_program = self.combine_program_with_image(inputs)

        return (
            self.model(inputs_combined_with_program, training=False)
            if self.model
            else inputs_combined_with_program
        )


def build_program_model(core_model=None, input_value_range=1.0, image_size=1.0):
    """Returns a model that takes an input, combines it with an adversarial program,
    and feeds the resulting image to another model.

    Args:
        core_model (tf.keras.Model, optional): The model into which the combination
                                               of the input and adversarial program is
                                               fed. Defaults to None.
        input_value_range (float, optional): A value between 0 and 1 indicating how
                                             the pixel values of the input are scaled.
                                             If 0, the input is ignored and only the
                                             adversarial program is fed into core_model.
                                             If 1, in those pixels that contain the
                                             input image, there is no adversarial
                                             program used. Defaults to 0.5.
        image_size (float, optional): The size of the image relative to the input size
                                     expected by core_model. If core_model is None,
                                     full image size is assumed to be 224x224.
                                     If image_size is given, the expected input shape
                                     of core_model has to be a square. Defaults to 1.

    Returns:
        tf.keras.Model: The desired model.
    """

    if core_model:
        core_model.trainable = False
        input_shape = find_input_shape(core_model)
    else:
        input_shape = (224, 224, 3)

    if input_shape[0] != input_shape[1]:
        raise ValueError("Input shape is not a square.")

    total_padding = input_shape[0] - round(image_size * input_shape[0])
    padding = (
        (math.floor(total_padding / 2), math.ceil(total_padding / 2)),
        (math.floor(total_padding / 2), math.ceil(total_padding / 2)),
    )

    ((top_pad, bottom_pad), (left_pad, right_pad)) = padding

    desired_image_height = input_shape[0] - top_pad - bottom_pad
    desired_image_width = input_shape[1] - left_pad - right_pad

    # create a mask containing 1 where the input image is and 0 everywhere else
    mask = tf.keras.layers.ZeroPadding2D(padding=padding)(
        tf.ones([1, desired_image_height, desired_image_width, input_shape[2]])
    )

    i = tf.keras.Input(shape=(None, None, input_shape[2]), name="image")

    # the input image is resized to the desired size
    x = tf.keras.layers.Resizing(
        desired_image_height, desired_image_width, interpolation="bilinear"
    )(i)

    # if necessary, pad the resized image further
    # so that we match the size expected by the network
    if top_pad or bottom_pad or left_pad or right_pad:
        x = tf.keras.layers.ZeroPadding2D(padding=padding)(x)

    x = AdditiveProgram(
        input_value_range=input_value_range, model=core_model, image_mask=mask
    )(x)

    return tf.keras.Model(inputs=[i], outputs=[x])


def networks(name="inception_v3", weights=None):
    """Returns the named network combined with the appropriate preprocessing layer."""

    if name == "inception_v3":
        preprocess_layer_callable = tf.keras.applications.inception_v3.preprocess_input
        core = tf.keras.applications.InceptionV3(weights=weights, include_top=True)
    elif name == "ResNet101V2":
        preprocess_layer_callable = tf.keras.applications.resnet_v2.preprocess_input
        core = tf.keras.applications.resnet_v2.ResNet101V2(
            weights=weights, include_top=True
        )
    elif name == "ResNet152V2":
        preprocess_layer_callable = tf.keras.applications.resnet_v2.preprocess_input
        core = tf.keras.applications.resnet_v2.ResNet152V2(
            weights=weights, include_top=True
        )
    elif name == "ResNet50V2":
        preprocess_layer_callable = tf.keras.applications.resnet_v2.preprocess_input
        core = tf.keras.applications.resnet_v2.ResNet50V2(
            weights=weights, include_top=True
        )
    elif name == "EfficientNetB0":
        preprocess_layer_callable = tf.keras.applications.efficientnet.preprocess_input
        core = tf.keras.applications.EfficientNetB0(weights=weights, include_top=True)
    elif name == "resnet50":
        preprocess_layer_callable = tf.keras.applications.resnet50.preprocess_input
        core = tf.keras.applications.ResNet50(weights=weights, include_top=True)
    else:
        raise ValueError(f"Unknown network name: {name}")

    input_shape = find_input_shape(core)

    i = tf.keras.Input(shape=input_shape, name="image")
    # include preprocessing layer just in case because some older model implementations
    # in keras required it and it does not hurt to include it in any case
    x = preprocess_layer_callable(i)
    x = core(x)
    return tf.keras.Model(inputs=[i], outputs=[x])


def get_network(
    network="inception_v3",
    input_value_range=1.0,
    image_size=1.0,
    learning_rate=0.01,
):
    """Returns a model with random weights based on the specified architecture.
    The model also contains a layer that adds an adversarial program to the input.

    Args:
        name (str, optional): The name of the base model as a string or a
                              tf.keras.Model instance. If a tf.keras.Model instance is
                              provided this is used as is including
                              its current weights whatever they may be.
                              If a string is given it can be one of inception_v3,
                              ResNet101V2, ResNet152V2, ResNet50V2, EfficientNetB0,
                              or resnet50. Defaults to 'inception_v3'.
        input_value_range (float, optional): Value between 0 and 1 indicating by how
                                             much the pixel range of the adversarial
                                             program is reduced. 1 means no adversarial
                                             program is added where the program and
                                             image overlap. 0 means the image is
                                             overwritten by the program.
                                             Defaults to 1.0.
        image_size (float, optional): Value between 0 and 1 indicating how large the
                                      image is relative to the program. 1 means the
                                      image is as large as the program. 0 means there
                                      is no image. Defaults to 1.0.
        learning_rate (float, optional): The learning rate for the Adam optimizer.
                                         Defaults to 0.01.
    """

    distribution_strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )

    if isinstance(network, tf.keras.Model):
        distribution_strategy = network.distribute_strategy
        model_pure = network
    else:
        with distribution_strategy.scope():
            model_pure = networks(name=network)
            model_pure = condition_batchnormalization_layers(model_pure, batch_size=50)

    with distribution_strategy.scope():
        model = build_program_model(
            core_model=model_pure,
            input_value_range=input_value_range,
            image_size=image_size,
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["categorical_accuracy"],
        )
    return model


def get_inputs(dataset_name="mnist", batch_size=50, classes=1000):
    """Returns a tuple of training and test datasets.
    Images are convertedto rgb and labels are one-hot encoded."""

    def reformat(x, y):
        # always return color images
        rgb_image = tf.image.grayscale_to_rgb(x) if x.shape[-1] == 1 else x
        # transform y into one hot encoding
        return rgb_image, tf.one_hot(y, depth=classes)

    ds = tfds.load(dataset_name, split="train", shuffle_files=True, as_supervised=True)
    ds_test = tfds.load(dataset_name, split="test", as_supervised=True)

    ds = ds.map(
        reformat, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True
    )
    ds = ds.shuffle(60000, reshuffle_each_iteration=True).batch(batch_size)

    ds_test = ds_test.map(
        reformat, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=True
    )
    ds_test = ds_test.batch(batch_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    return ds.with_options(options), ds_test.with_options(options)


def run_experiment(
    network,
    dataset_name="mnist",
    input_value_range=1.0,
    image_size=1.0,
    epochs=20,
    learning_rate=0.01,
    batch_size=50,
):
    """Run the experiment for the specified parameters. Returns the tf.keras.callbacks.
    History instance generated during training as well as the base model (including all
    the weights). The latter is returned so that additional experiments can be run with
    exactly the same network if so desired."""

    ds, ds_test = get_inputs(dataset_name=dataset_name, batch_size=batch_size)
    model = get_network(
        network,
        input_value_range,
        image_size,
        learning_rate=learning_rate,
    )
    print(
        f"Running experiment for {model.layers[-1].model.layers[-1].name} and "
        f"{dataset_name} with input_value_range {input_value_range} and image_size "
        f"{image_size} for {epochs} epochs using a learning rate of {learning_rate} "
        f"and batch size {batch_size}"
    )
    history = model.fit(ds, epochs=epochs, validation_data=ds_test, verbose=2)
    print(
        f"{model.layers[-1].model.layers[-1].name} - {dataset_name} - "
        f"{input_value_range} - {image_size} - {epochs} - {learning_rate} - "
        f"{batch_size}: {history.history['val_categorical_accuracy'][-1]}"
    )
    return history, model.layers[-1].model


if __name__ == "__main__":

    def float_range(x):
        """Raise exception if x isn't a float between 0 and 1."""
        try:
            x = float(x)
        except ValueError:
            # pylint: disable-next=raise-missing-from
            raise argparse.ArgumentTypeError(f"{x} is not a number")
        if x < 0.0 or x > 1.0:
            raise argparse.ArgumentTypeError(f"{x} is not between 0.0 and 1.0")
        return x

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        metavar="N",
        choices=[
            "resnet50",
            "ResNet101V2",
            "ResNet50V2",
            "ResNet152V2",
            "EfficientNetB0",
            "inception_v3",
        ],
        required=True,
    )
    parser.add_argument(
        "--dataset",
        metavar="D",
        choices=["mnist", "fashion_mnist", "kmnist"],
        default="mnist",
    )
    parser.add_argument(
        "--input_value_range", metavar="V", type=float_range, default=1.0
    )
    parser.add_argument("--image_size", metavar="S", type=float_range, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", help="learning rate", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=50)
    args = parser.parse_args()

    run_experiment(
        network=args.network,
        dataset_name=args.dataset,
        input_value_range=args.input_value_range,
        image_size=args.image_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
