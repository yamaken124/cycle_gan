import argparse
import os, sys
import numpy as np
import datetime
import time
import pickle
import random
import chainer
from chainer import cuda
from chainer import serializers
import chainer.functions as F
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import Generator, Discriminator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import ImageDataset


def visualize(genA, genB, realA, realB, savedir):
    img_realA = ((realA + 1) * 127.5).clip(0, 255).astype(np.uint8)
    with chainer.using_config('train', True), chainer.no_backprop_mode():
        x_fakeB = genB(chainer.Variable(genB.xp.asarray(realA, 'float32')))
        x_recA = genA(x_fakeB)
    img_fakeB = ((cuda.to_cpu(x_fakeB.data) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img_recA = ((cuda.to_cpu(x_recA.data) + 1) * 127.5).clip(0, 255).astype(np.uint8)

    img_realB = ((realB + 1) * 127.5).clip(0, 255).astype(np.uint8)
    with chainer.using_config('train', True), chainer.no_backprop_mode():
        x_fakeA = genA(chainer.Variable(genA.xp.asarray(realB, 'float32')))
        x_recB = genB(x_fakeA)
    img_fakeA = ((cuda.to_cpu(x_fakeA.data) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img_recB = ((cuda.to_cpu(x_recB.data) + 1) * 127.5).clip(0, 255).astype(np.uint8)

    # for i in range(10):
    #     save_file = savedir + '/' + str(i) + ".jpg"
    #     print(save_file)
    #     cv2.imwrite(save_file, img_fakeB[i].transpose(1,2,0))
    # print(img_fakeB[0].transpose(1,2,0).shape())
    fig = plt.figure(figsize=(10, 30))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(75):
        ax = fig.add_subplot(15, 5, i + 1, xticks=[], yticks=[])
        # row1
        if i < 5:
            ax.imshow(img_realA[i].transpose(1, 2, 0))
        elif i < 10:
            ax.imshow(img_fakeB[i - 5].transpose(1, 2, 0))
        elif i < 15:
            ax.imshow(img_recA[i - 10].transpose(1, 2, 0))
        # row2
        elif i < 20:
            ax.imshow(img_realA[i - 10].transpose(1, 2, 0))
        elif i < 25:
            ax.imshow(img_fakeB[i - 15].transpose(1, 2, 0))
        elif i < 30:
            ax.imshow(img_recA[i - 20].transpose(1, 2, 0))
        # row3
        elif i < 35:
            ax.imshow(img_realA[i - 20].transpose(1, 2, 0))
        elif i < 40:
            ax.imshow(img_fakeB[i - 25].transpose(1, 2, 0))
        elif i < 45:
            ax.imshow(img_recA[i - 30].transpose(1, 2, 0))
        # row4
        elif i < 50:
            ax.imshow(img_realA[i - 30].transpose(1, 2, 0))
        elif i < 55:
            ax.imshow(img_fakeB[i - 35].transpose(1, 2, 0))
        elif i < 60:
            ax.imshow(img_recA[i - 40].transpose(1, 2, 0))
        # row5
        elif i < 65:
            ax.imshow(img_realA[i - 40].transpose(1, 2, 0))
        elif i < 70:
            ax.imshow(img_fakeB[i - 45].transpose(1, 2, 0))
        elif i < 75:
            ax.imshow(img_recA[i - 50].transpose(1, 2, 0))

    plt.savefig('{}/samples'.format(savedir))
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU device ID')
    parser.add_argument('--epoch', '-e', type=int, default=200, help='# of epoch')
    parser.add_argument('--batch_size', '-b', type=int, default=5)
    parser.add_argument('--memory_size', '-m', type=int, default=200)
    parser.add_argument('--real_label', type=float, default=0.9)
    parser.add_argument('--fake_label', type=float, default=0.0)
    parser.add_argument('--block_num', type=int, default=5)
    parser.add_argument('--g_nobn', dest='g_bn', action='store_false', default=True)
    parser.add_argument('--d_nobn', dest='d_bn', action='store_false', default=True)
    parser.add_argument('--variable_size', action='store_true', default=False)
    parser.add_argument('--lambda_dis_real', type=float, default=0)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--lambda_', type=float, default=10)
    parser.add_argument('--out', '-o', type=str, default='output') # output dir

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # log directory
    out = datetime.datetime.now().strftime('%m%d%H')
    out = out + '_' + args.out
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", out))
    os.makedirs(os.path.join(out_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'visualize'), exist_ok=True)

    # hyper parameter
    with open(os.path.join(out_dir, 'setting.txt'), 'w') as f:
        for k, v in args._get_kwargs():
            print('{} = {}'.format(k, v))
            f.write('{} = {}\n'.format(k, v))

    # genA convert B -> A, genB convert A -> B
    genA = Generator(block_num=args.block_num, bn=args.g_bn)
    genB = Generator(block_num=args.block_num, bn=args.g_bn)
    # disA discriminate realA and fakeA, disB discriminate realB and fakeB
    disA = Discriminator(bn=args.d_bn)
    disB = Discriminator(bn=args.d_bn)

    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        genA.to_gpu()
        genB.to_gpu()
        disA.to_gpu()
        disB.to_gpu()

    valA = ImageDataset('black2blond/valA', image_size=178, final_size=args.size)
    # valA = ImageDataset('black2blond/testA', image_size=178, final_size=args.size)
    valB = ImageDataset('black2blond/testB', image_size=178, final_size=args.size)
    const_valA = np.asarray([valA.get_example(i) for i in range(25)])
    const_valB = np.asarray([valB.get_example(i) for i in range(25)])

    # [TODO] replace with your pass
    serializers.load_hdf5(os.path.join("./runs/output/", "models", "{:03d}.disA.model".format(195)), disA)
    serializers.load_hdf5(os.path.join("./runs/output/", "models", "{:03d}.disB.model".format(195)), disB)
    serializers.load_hdf5(os.path.join("./runs/output/", "models", "{:03d}.genA.model".format(195)), genA)
    serializers.load_hdf5(os.path.join("./runs/output/", "models", "{:03d}.genB.model".format(195)), genB)

    visualize(genA, genB, const_valA, const_valB, savedir=os.path.join(out_dir, 'visualize'))


if __name__ == '__main__':
    main()
