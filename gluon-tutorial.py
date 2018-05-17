#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by C. L. Wang on 2018/5/15
"""
from time import time

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import MNIST
from mxnet.io import NDArrayIter
from mxnet.test_utils import get_mnist


class GluonFirst(object):

    def __init__(self):
        self.data_ctx = mx.cpu()
        self.model_ctx = mx.cpu()

        GPU_COUNT = 4
        ctx = [mx.gpu(i) for i in range(GPU_COUNT)]
        self.model_ctx_gpu = ctx  # GPU

        self.batch_size = 64
        self.num_outputs = 10
        pass

    def load_data(self):
        """
        加载MNIST数据集
        """

        def transform(data, label):
            return data.astype(np.float32) / 255., label.astype(np.float32)

        # 训练和测试数据
        train_data = DataLoader(MNIST(train=True, transform=transform),
                                self.batch_size, shuffle=True)
        valid_data = DataLoader(MNIST(train=False, transform=transform),
                                self.batch_size, shuffle=False)
        # mnist = get_mnist()
        # train_data = NDArrayIter(mnist["train_data"], mnist["train_label"], self.batch_size)
        # valid_data = NDArrayIter(mnist["test_data"], mnist["test_label"], self.batch_size)

        return train_data, valid_data

    def model(self):
        num_hidden = 64
        net = gluon.nn.Sequential()
        with net.name_scope():
            net.add(gluon.nn.Dense(units=num_hidden, activation="relu"))
            net.add(gluon.nn.Dense(units=num_hidden, activation="relu"))
            net.add(gluon.nn.Dense(units=self.num_outputs))

        print(net)  # 展示模型
        return net

    def __evaluate_accuracy(self, data_iterator, net):
        acc = mx.metric.Accuracy()  # 准确率
        for i, (data, label) in enumerate(data_iterator):
            data = data.as_in_context(self.model_ctx).reshape((-1, 784))
            label = label.as_in_context(self.model_ctx)
            output = net(data)  # 预测结果
            predictions = nd.argmax(output, axis=1)  # 类别
            acc.update(preds=predictions, labels=label)  # 更新概率和标签
        return acc.get()[1]  # 第1维是数据名称，第2位是概率

    def train(self):
        train_data, test_data = self.load_data()  # 训练和测试数据
        net = self.model()  # 模型
        net.collect_params().initialize(init=mx.init.Normal(sigma=.1), ctx=self.model_ctx)

        epochs = 10
        smoothing_constant = .01
        num_examples = 60000

        softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()  # 交叉熵
        trainer = gluon.Trainer(params=net.collect_params(),
                                optimizer='sgd',
                                optimizer_params={'learning_rate': smoothing_constant})

        for e in range(epochs):
            cumulative_loss = 0  # 累积的
            for i, (data, label) in enumerate(train_data):
                data = data.as_in_context(self.model_ctx).reshape((-1, 784))  # 数据
                label = label.as_in_context(self.model_ctx)  # 标签

                with autograd.record():  # 梯度
                    output = net(data)  # 输出
                    loss = softmax_cross_entropy(output, label)  # 输入和输出计算loss

                loss.backward()  # 反向传播
                trainer.step(data.shape[0])  # 设置trainer的step
                cumulative_loss += nd.sum(loss).asscalar()  # 计算全部损失

            test_accuracy = self.__evaluate_accuracy(test_data, net)
            train_accuracy = self.__evaluate_accuracy(train_data, net)
            print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
                  (e, cumulative_loss / num_examples, train_accuracy, test_accuracy))

    def train_gpu(self):
        train_data, valid_data = self.load_data()  # 训练和测试数据
        ctx = self.model_ctx_gpu
        print('Running on {}'.format(ctx))

        net = self.model()  # 模型
        net.collect_params().initialize(init=mx.init.Normal(sigma=.1), ctx=ctx)

        smoothing_constant = .01
        trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': smoothing_constant})

        epochs = 10
        for e in range(epochs):
            start = time()
            for batch in train_data:
                self.train_batch(batch, ctx, net, trainer)
            nd.waitall()  # 等待所有异步的任务都终止
            print('Epoch %d, training time = %.1f sec' % (e, time() - start))
            correct, num = 0.0, 0.0
            for batch in valid_data:
                correct += self.valid_batch(batch, ctx, net)
                num += batch[0].shape[0]
            print('\tvalidation accuracy = %.4f' % (correct / num))

    @staticmethod
    def train_batch(batch, ctx, net, trainer):
        # split the data batch and load them on GPUs
        data = gluon.utils.split_and_load(batch[0], ctx)  # 列表
        label = gluon.utils.split_and_load(batch[1], ctx)  # 列表
        # compute gradient
        GluonFirst.forward_backward(net, data, label)
        # update parameters
        trainer.step(batch[0].shape[0])

    @staticmethod
    def valid_batch(batch, ctx, net):
        data = batch[0].as_in_context(ctx[0])
        pred = nd.argmax(net(data), axis=1)
        return nd.sum(pred == batch[1].as_in_context(ctx[0])).asscalar()

    @staticmethod
    def forward_backward(net, data, label):
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        with autograd.record():
            losses = [loss(net(X), Y) for X, Y in zip(data, label)]  # loss列表
        for l in losses:  # 每个loss反向传播
            l.backward()


if __name__ == '__main__':
    gf = GluonFirst()
    gf.train()
