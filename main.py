""" Estimator API for CNNs using popular implementations """
import os
import random

import tensorflow as tf
import numpy as np

from estimator import model_fn


#################################
# Parameters
#################################

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    'root_path',
    '',
    "Images root path - must contain directories with class specific images")

flags.DEFINE_string(
    'model_save_path', '',
    "Path in which to save graphs, models and summaries")

flags.DEFINE_string(
    'model', 'small_cnn',
    "Model name")

flags.DEFINE_integer(
    'max_epoch', 10,
    "Max epoch to train model")

flags.DEFINE_integer(
    'batch_size', 64,
    "Batch size for model training")

flags.DEFINE_integer(
    'image_size', 50,
    "Image size (width/height) for model input")

flags.DEFINE_integer(
    'num_gpus', 0,
    "Number of GPUs for model training")

flags.DEFINE_integer(
    'num_cpus', 2,
    "Numer of CPUs (for pre-processing)")

flags.DEFINE_float('train_fraction', 0.8, "training set fraction")

flags.DEFINE_bool(
    'color_augmentation', True,
    "Whether to randomly adjust colors during model training")

flags.DEFINE_float(
    'weight_decay', 0,
    'Applies weight decay if supported by specific model')

flags.DEFINE_list(
    'image_means', [0, 0, 0],
    'image means (leave at default for automatic mode)')

flags.DEFINE_list(
    'image_stdevs', [1, 1, 1],
    'image stdevs (leave at default for automatic mode)')


#################################
# Define Dataset
#################################

# get all class directories
classes = os.listdir(FLAGS.root_path)
n_classes = len(classes)

# find all images
image_paths = dict()
for cl in classes:
    image_names = os.listdir(os.path.join(FLAGS.root_path, cl))
    image_paths[cl] = [os.path.join(FLAGS.root_path, cl, x)
                       for x in image_names]

# Map classes to numerics
classes_to_num_map = {k: i for i, k in enumerate(classes)}
num_to_class_map = {v: k for k, v in classes_to_num_map.items()}

# Create lists of image paths and labels
label_list = list()
image_path_list = list()
for k, v in image_paths.items():
    label_list += [classes_to_num_map[k] for i in range(0, len(v))]
    image_path_list += v

# randomly shuffle input to ensure good mixing when model training
indices = [i for i in range(0, len(label_list))]
random.seed(123)
random.shuffle(indices)
image_path_list = [image_path_list[i] for i in indices]
label_list = [label_list[i] for i in indices]

n_records = len(label_list)

# Create training and test set
train_fraction = FLAGS.train_fraction
n_train = int(round(n_records * train_fraction, 0))
n_test = n_records - n_train

train_files = image_path_list[0: n_train]
train_labels = label_list[0: n_train]

test_files = image_path_list[n_train:]
test_labels = label_list[n_train:]

#################################
# Dataset Iterator
#################################

# build tensors with filenames and labels
filenames_train = tf.constant(train_files)
labels_train = tf.constant(train_labels)

filenames_test = tf.constant(test_files)
labels_test = tf.constant(test_labels)


# Standardize a single image
def _standardize_images(image, means, stdevs):
    """ Standardize images """
    with tf.name_scope("image_standardization"):
        means = tf.expand_dims(tf.expand_dims(means, 0), 0)
        means = tf.cast(means, tf.float32)

        stdevs = tf.expand_dims(tf.expand_dims(stdevs, 0), 0)
        stdevs = tf.cast(stdevs, tf.float32)

        image = image - means
        image = tf.divide(image, stdevs)

        return image


# data augmentation
def _image_augmentation(image):
    """ Apply some random image augmentation """
    with tf.name_scope("image_augmentation"):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.9, upper=1)
        image = tf.image.random_hue(image, max_delta=0.02)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

        return image


# parse a single image
def _parse_function(filename, label, augmentation=True):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize_images(image, [FLAGS.image_size, FLAGS.image_size])
    image = tf.divide(image, 255.0)
    if augmentation:
        image = _image_augmentation(image)
    image = _standardize_images(image, FLAGS.image_means,
                                FLAGS.image_stdevs)
    return image, label


def dataset_iterator(filenames, labels, is_train, augmentation=True):
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(lambda x, y: _parse_function(x, y, augmentation),
                          num_parallel_calls=FLAGS.num_cpus)
    if is_train:
        dataset = dataset.shuffle(buffer_size=300)
    dataset = dataset.batch(FLAGS.batch_size)
    if is_train:
        dataset = dataset.repeat(1)
    else:
        dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    features = {'images': images, 'labels': labels}
    return features


# Create callable iterator functions
def train_iterator():
    return dataset_iterator(filenames_train, labels_train, True,
                            FLAGS.color_augmentation)


def test_iterator():
    return dataset_iterator(filenames_test, labels_test, False, False)


def original_iterator():
    return dataset_iterator(filenames_train, labels_train, False, False)


#################################
# Image Statistics for Preprocessing
#################################


# Calculate image means and stdevs of training images for RGB channels
# for image standardization

if (FLAGS.image_means == [0, 0, 0]) and (FLAGS.image_stdevs == [1, 1, 1]):
    with tf.Session() as sess:
        original_batch_size = FLAGS.batch_size
        FLAGS.batch_size = np.min([500, n_train])
        features = sess.run(original_iterator())
        image_batch = features['images']
        means_batch = np.mean(image_batch, axis=(0, 1, 2))
        stdev_batch = np.std(image_batch, axis=(0, 1, 2))
        FLAGS.batch_size = original_batch_size

    image_means = [round(float(x), 6) for x in list(means_batch)]
    image_stdevs = [round(float(x), 4) for x in list(stdev_batch)]

    FLAGS.image_means = image_means
    FLAGS.image_stdevs = image_stdevs


#################################
# Configure Estimator
#################################


n_batches_per_epoch_train = int(round(n_train / FLAGS.batch_size))

# Configurations
config_sess = tf.ConfigProto()
config_sess.gpu_options.per_process_gpu_memory_fraction = 0.8
config_sess.gpu_options.allow_growth = True


def distribution_gpus(num_gpus):
    if num_gpus == 0:
        return tf.contrib.distribute.OneDeviceStrategy(device='/cpu:0')
    elif num_gpus == 1:
        return tf.contrib.distribute.OneDeviceStrategy(device='/gpu:0')
    elif num_gpus > 1:
        return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
    else:
        return None


# Config estimator
est_config = tf.estimator.RunConfig()
est_config = est_config.replace(
        keep_checkpoint_max=3,
        save_checkpoints_steps=n_batches_per_epoch_train,
        session_config=config_sess,
        save_checkpoints_secs=None,
        save_summary_steps=n_batches_per_epoch_train,
        model_dir=FLAGS.model_save_path,
        train_distribute=distribution_gpus(FLAGS.num_gpus))

# Model Parameters
params = dict()
params['label'] = ['labels']
params['n_classes'] = [n_classes]
params['weight_decay'] = FLAGS.weight_decay
params['momentum'] = 0.9
params['model'] = FLAGS.model
params['reuse'] = False
params['class_mapping_clean'] = {'labels': num_to_class_map}

# create estimator
estimator = tf.estimator.Estimator(model_fn=model_fn,
                                   params=params,
                                   model_dir=FLAGS.model_save_path,
                                   config=est_config
                                   )

#################################
# Train and Evaluate
#################################


def main(args):
    """ Main - called by command line """

    # Print flags
    for f in flags.FLAGS:
        print("Flag %s - %s" % (f, FLAGS[f].value))

    eval_loss = list()

    for epoch in range(1, FLAGS.max_epoch + 1):
        print("Starting with epoch %s" % epoch)

        # Train for one epoch
        estimator.train(input_fn=train_iterator)

        # Evaluate
        eval_res = estimator.evaluate(input_fn=test_iterator)
        print("Evaluation results:")
        for k, v in eval_res.items():
            print("   Res for %s - %s" % (k, v))

        eval_loss.append(eval_res['loss'])

    # Predict
    preds = estimator.predict(input_fn=test_iterator)

    for i, pred in enumerate(preds):
        print(pred)
        if i > 10:
            break


if __name__ == '__main__':
    tf.app.run()
