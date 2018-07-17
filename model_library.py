""" Define a collection of models, imported from different sources:
    - TF-slim
    - Tensornets
    - Own models
"""
import tensorflow as tf
import models

try:
    import tensornets
except ImportError:
    tensornets = models.tensornets

from models.resnet_official import resnet_model
from models.slim.nets.inception_resnet_v2 import inception_resnet_v2
from models.slim.nets.nets_factory import get_network_fn
from models.small_cnn import small_cnn
from tensornets.mobilenets import mobilenet25


class Model(object):
    """ A Model """

    models = {}

    def get_output(self):
        """ Return Output """
        raise NotImplementedError

    @classmethod
    def register_model(mod, model_name):
        def decorator(model):
            mod.models[model_name] = model
            return model
        return decorator

    @classmethod
    def create(mod, model_name, input, training, params):
        if model_name not in mod.models:
            raise ValueError('Bad model name {}'.format(model_name))

        return mod.models[model_name](input, training, params)


#################################
# mobilenet25 (tensornets)
#################################

@Model.register_model('mobilenet25')
class Mobilenet25(Model):

    def __init__(self, input, training, params=None):
        self.input = input
        self.training = training
        self.mod = mobilenet25(
            self.input, is_training=training, classes=2, stem=True,
            reuse=params['reuse'])
        self.outputs = tf.reduce_mean(self.mod, [1, 2], name='avgpool')
        #self.outputs = tf.squeeze(self.outputs, axis=(1, 2))

    def get_output(self):
        """ Return Output Layer """
        return self.outputs

#################################
# ResNet
#################################


@Model.register_model('ResNet18')
class ResNet18(Model):

    def __init__(self, input, training, params=None):
        self.input = input
        self.training = training
        self.params = params
        self.mod = _get_resnet_model(18)

    def get_output(self):
        """ Return Output Layer """
        outputs = self.mod(self.input, self.training)
        _add_resnet_regularization_loss(self.params['weight_decay'])
        return outputs


@Model.register_model('ResNet32')
class ResNet32(Model):

    def __init__(self, input, training, params=None):
        self.input = input
        self.training = training
        self.mod = _get_resnet_model(32)
        _add_resnet_regularization_loss(params['weight_decay'])

    def get_output(self):
        """ Return Output Layer """
        return self.mod(self.input, self.training)


@Model.register_model('ResNet50')
class ResNet50(Model):

    def __init__(self, input, training, params=None):
        self.input = input
        self.training = training
        self.mod = _get_resnet_model(50)
        _add_resnet_regularization_loss(params['weight_decay'])

    def get_output(self):
        """ Return Output Layer """
        return self.mod(self.input, self.training)


@Model.register_model('ResNet101')
class ResNet101(Model):

    def __init__(self, input, training, params=None):
        self.input = input
        self.training = training
        self.mod = _get_resnet_model(101)
        _add_resnet_regularization_loss(params['weight_decay'])

    def get_output(self):
        """ Return Output Layer """
        return self.mod(self.input, self.training)


@Model.register_model('ResNet152')
class ResNet152(Model):

    def __init__(self, input, training, params=None):
        self.input = input
        self.training = training
        self.mod = _get_resnet_model(152)
        _add_resnet_regularization_loss(params['weight_decay'])

    def get_output(self):
        """ Return Output Layer """
        return self.mod(self.input, self.training)


def _add_resnet_regularization_loss(weight_decay=0.0):
    """ Add regularization loss to collection """
    l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])
    tf.losses.add_loss(
        l2_loss,
        loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)


def _get_resnet_model(size):
    """ Helper Function to Create a ResNet Model """

    resnet_size = size
    if resnet_size < 50:
        bottleneck = False
        final_size = 512
    else:
        bottleneck = True
        final_size = 2048

    mod = resnet_model.Model(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=None,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=resnet_model._get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        resnet_version=resnet_model.DEFAULT_VERSION,
        data_format=None,
        dtype=resnet_model.DEFAULT_DTYPE
    )

    return mod


#################################
# MobileNet V1
#################################

@Model.register_model('MobileNetV1')
class MobileNetV1(Model):

    def __init__(self, input, training, params=None):

        self.net_fun = get_network_fn(
            name='mobilenet_v1_025',
            num_classes=0,
            weight_decay=params['weight_decay'],
            is_training=training)

        pool, outputs = self.net_fun(
            images=input,
            global_pool=True,
            reuse=params['reuse']
            )

        self.outputs = tf.squeeze(pool, axis=(1, 2))

    def get_output(self):
        """ Return Output Layer """
        return self.outputs

#################################
# alexnet
#################################


@Model.register_model('alexnet')
class Alexnet(Model):

    def __init__(self, input, training, params=None):

        self.net_fun = get_network_fn(
            name='alexnet_v2',
            num_classes=0,
            weight_decay=params['weight_decay'],
            is_training=training)

        pool, outputs = self.net_fun(
            images=input,
            global_pool=True
            )

        self.outputs = tf.squeeze(pool, axis=(1, 2))

    def get_output(self):
        """ Return Output Layer """
        return self.outputs

#################################
# InceptionResNetV2
#################################


@Model.register_model('InceptionResNetV2')
class InceptionResNetV2(Model):

    def __init__(self, input, training, params=None):

        net, end_points = inception_resnet_v2(
            input, num_classes=None,
            is_training=training,
            dropout_keep_prob=0.8,
            reuse=None,
            scope='InceptionResnetV2',
            create_aux_logits=True,
            activation_fn=tf.nn.relu)

        self.outputs = net

    def get_output(self):
        """ Return Output Layer """
        return self.outputs


#################################
# SmallCNN
#################################


@Model.register_model('small_cnn')
class SmallCNN(Model):

    def __init__(self, input, training, params=None):
        self.outputs = small_cnn(input, training, reuse=params['reuse'],
                                 weight_decay=params['weight_decay'])

    def get_output(self):
        """ Return Output Layer """
        return self.outputs
