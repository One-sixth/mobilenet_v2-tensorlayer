import tensorflow as tf
import tensorlayer as tl
import numpy as np
from progressbar import progressbar
import like_mobilenet_v2

x_train, y_train, x_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))


img = tf.placeholder(tf.float32, (None, 32, 32, 3))
label = tf.placeholder(tf.int64, (None,))
# img_scale = tf.image.resize_images(img, [224, 224])

# net, l2_loss = like_mobilenet_v2.get_model(img_scale, 10, True, False)
net, l2_loss = like_mobilenet_v2.get_model(img, 10, True, False)

cost = tf.losses.sparse_softmax_cross_entropy(label, net.outputs)

correct_prediction = tf.equal(tf.argmax(net.outputs, 1), label)
acc_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_op = tf.train.AdamOptimizer(0.001).minimize(cost, var_list=net.all_params)

sess = tf.Session()
tl.layers.initialize_global_variables(sess)

epoch = 200
batch_size = 500
train_batch_count = int(np.ceil(len(x_train) / batch_size))
test_batch_count = int(np.ceil(len(x_test) / batch_size))

for e in range(epoch):
    loss = 0
    for b in progressbar(range(train_batch_count)):
        feed_dict = {img:x_train[b*batch_size : (b+1)*batch_size], label:y_train[b*batch_size : (b+1)*batch_size]}
        los, _ = sess.run([cost, train_op], feed_dict)
        loss += los
    print('train loss', loss / train_batch_count)
    if e % 5 == 1:
        loss = 0
        acc = 0
        for b in progressbar(range(test_batch_count)):
            feed_dict = {img: x_test[b * batch_size: (b + 1) * batch_size],
                         label: y_test[b * batch_size: (b + 1) * batch_size]}
            los, ac = sess.run([cost, acc_op], feed_dict)
            loss += los
            acc += ac
        print('test loss', loss / test_batch_count, 'test acc', acc / test_batch_count)
