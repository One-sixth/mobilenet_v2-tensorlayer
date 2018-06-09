import tensorflow as tf
import tensorlayer as tl
import numpy as np

def get_model(x, k=None, is_train=True, reuse=False):
    act = tf.nn.leaky_relu
    with tf.variable_scope('mobilenet_v2', reuse=reuse) as vs:
        def bottleneck(x, n_filter, stride=1, t=6, name='bt'):
            # tl.layers.SeparableConv2d(x, n_filter, (3, 3), stride, act, 'same')
            net = tl.layers.Conv2d(x, n_filter * t, (1, 1), b_init=None, name='%s_expand' % name)
            net = tl.layers.BatchNormLayer(net, act=act, is_train=is_train, name='%s_bn1' % name)
            net = tl.layers.DepthwiseConv2d(net, (3, 3), (stride, stride), b_init=None, name='%s_dwise1' % name)
            net = tl.layers.BatchNormLayer(net, act=act, is_train=is_train, name='%s_bn2' % name)
            net = tl.layers.Conv2d(net, n_filter, (1, 1), b_init=None, name='%s_project' % name)
            net = tl.layers.BatchNormLayer(net, act=tf.identity, is_train=is_train, name='%s_bn3' % name)
            if stride == 1:
                last_n_filter = x.outputs.get_shape()[-1]
                if n_filter > last_n_filter:
                    shortcut = tl.layers.PadLayer(x, [[0,0],[0,0],[0,0],[0, n_filter-last_n_filter]])
                elif n_filter < last_n_filter:
                    shortcut = tl.layers.Conv2d(x, n_filter, 1, 1, name='%s_shortcut' % name)
                else:
                    shortcut = x
                net = tl.layers.ElementwiseLayer([net, shortcut], tf.add)
            return net

        # 224 x 224 x 3
        net = tl.layers.InputLayer(x)
        net = tl.layers.Conv2d(net, 32, (3, 3), (2, 2), b_init=None, name='conv1')
        net = tl.layers.BatchNormLayer(net, act=act, is_train=is_train, name='bn1')
        for i in range(1):
            stride = 1
            net = bottleneck(net, 16, stride, 1, 'bt0_%d' % i)
        for i in range(2):
            if i == 0:
                stride = 2
            else:
                stride = 1
            net = bottleneck(net, 24, stride, 6, 'bt1_%d' % i)
        for i in range(3):
            if i == 0:
                stride = 2
            else:
                stride = 1
            net = bottleneck(net, 32, stride, 6, 'bt2_%d' % i)
        for i in range(4):
            if i == 0:
                stride = 2
            else:
                stride = 1
            net = bottleneck(net, 64, stride, 6, 'bt3_%d' % i)
        for i in range(3):
            stride = 1
            net = bottleneck(net, 96, stride, 6, 'bt4_%d' % i)
        for i in range(3):
            if i == 0:
                stride = 2
            else:
                stride = 1
            net = bottleneck(net, 160, stride, 6, 'bt5_%d' % i)
        for i in range(1):
            stride = 1
            net = bottleneck(net, 320, stride, 6, 'bt6_%d' % i)

        if k != None:
            net = tl.layers.Conv2d(net, 1280, 1, 1, b_init=None, name='conv2')
            net = tl.layers.BatchNormLayer(net, act=act, is_train=is_train, name='bn2')
            net = tl.layers.MeanPool2d(net, net.outputs.get_shape()[1:3], 1, 'VALID', 'meanpool1')
            net = tl.layers.Conv2d(net, k, 1, 1, name='output')
            net = tl.layers.FlattenLayer(net)

        ws = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, vs.name + '.*kernel.*')
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in ws])

    return net, l2_loss

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, (None, 224, 224, 3))
    net, l2_loss = get_model(x, 1280, True, False)
    writer = tf.summary.FileWriter('log/', graph=tf.get_default_graph())
    writer.close()
    # net = tl.layers.SeparableConv2d(tl.layers.InputLayer(x), 100, padding='SAME')