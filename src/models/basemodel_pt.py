from layer_utils import *
from model import Model
from os.path import join
import numpy as np
import os
import tensorflow as tf
import cPickle


class BaseModel_pt(Model):

    def __init__(self, opts):

        B = opts.BATCH_SIZE
        H = opts.IN_HEIGHT
        W = opts.IN_WIDTH
        C = opts.IN_CHANNELS
        T = opts.NUM_JOINTS

        self.opts = opts

        self.x = tf.placeholder(tf.float32, [B, H, W, C], name="x")
        self.y = tf.placeholder(tf.float32, [B, T, 2], name="y")
        self.global_step = tf.Variable(0,trainable=False)

        c0_0 = conv2d_relu(self.x, [3, 3, C, 64], 1, "conv1_1")
        c0_1 = conv2d_relu(c0_0, [3, 3, 64, 64], 1, "conv1_2")
        p0 = maxpool2d(c0_1, 2, "pool1")

        c1_0 = conv2d_relu(p0, [3, 3, 64, 128], 1, "conv2_1")
        c1_1 = conv2d_relu(c1_0, [3, 3, 128, 128], 1, "conv2_2")
        p1 = maxpool2d(c1_1, 2, "pool2")

        c2_0 = conv2d_relu(p1, [3, 3, 128, 256], 1, "conv3_1")
        c2_1 = conv2d_relu(c2_0, [3, 3, 256, 256], 1, "conv3_2")
        c2_2 = conv2d_relu(c2_1, [3, 3, 256, 256], 1, "conv3_3")
        p2 = maxpool2d(c2_2, 2, "pool3")

        c3_0 = conv2d_relu(p2, [3, 3, 256, 512], 1, "conv4_1")
        c3_1 = conv2d_relu(c3_0, [3, 3, 512, 512], 1, "conv4_2")
        c3_2 = conv2d_relu(c3_1, [3, 3, 512, 512], 1, "conv4_3")
        p3 = maxpool2d(c3_2, 2, "pool4")

        c4_0 = conv2d_relu(p3, [3, 3, 512, 512], 1, "conv5_1")
        c4_1 = conv2d_relu(c4_0, [3, 3, 512, 512], 1, "conv5_2")
        c4_2 = conv2d_relu(c4_1, [3, 3, 512, 512], 1, "conv5_3")
        p4 = maxpool2d(c4_2, 2, "pool5")


        reshape = tf.reshape(p4, [B, -1])
        dim = reshape.get_shape()[1]
        fc5 = dense_relu_batch(reshape, dim, 4096, [0], 'fcl5')
        fc6 = dense_relu_batch(fc5, 4096,4096, [0], 'fcl6')
        self.logits = tf.reshape(dense(fc6, 4096, T*2, 'logits'), [B, T, 2])
        
        self.diff = (self.logits - self.y) * tf.to_float(self.y > 0)
        self.loss = tf.nn.l2_loss(self.diff)
        self.rmse = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.diff), 2)), 1))
        self.train_op = self.train_op_init(self.loss, self.global_step)

        self.saver = tf.train.Saver(tf.all_variables())
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.sess.run(self.pretrain_op())


    # returns list of tensors with assign ops
    def pretrain_op(self):
        datapath = join(self.opts.PRETRAINED_DIR, 'vgg16.pkl')
        with open(datapath, 'rb') as fid:
            weight_dict = cPickle.load(fid)
        print 'VGG16 loaded from {}'.format(datapath)
        pretrain_ops = []
        for var in tf.all_variables():
            tags = var.name.split('/')
            name = tags[0] + '_' + tags[-1].split(':')[0].lower()
            if name in weight_dict:
                print 'Pretrained match for : %s' % name
                pretrain_ops.append(var.assign(weight_dict[name]))
            elif 'conv' in var.name:
                print 'Pretrain match not found for : %s' % var.name
        return pretrain_ops



if __name__ == '__main__':
    # Basic functionality test
    import sys
    sys.path.append("..")
    from GlobalOpts import GlobalOpts
    opts = GlobalOpts('basemodel')

    model = BaseModel(opts)
    for i in range(1):
        batchX = np.random.rand(opts.BATCH_SIZE, opts.IN_HEIGHT, opts.IN_WIDTH, 3)
        batchy = np.random.rand(opts.BATCH_SIZE, opts.NUM_JOINTS, 2)
        loss = model.train_step(batchX, batchy)
        print loss
        pred = model.predict(batchX)
        print np.array(pred).shape
