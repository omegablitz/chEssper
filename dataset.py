import os
import numpy


def load_emnist(dim=2, norm=True, DATA_ROOT='binary'):

    cache_path = '.cache/emnist.{}{}.npz'.format(dim, '.norm' if norm else '')

    if os.path.exists(cache_path):
        data = numpy.load(cache_path)
        images_train = data['images_train']
        labels_train = data['labels_train']
        images_test = data['images_test']
        labels_test = data['labels_test']
        return images_train, labels_train, images_test, labels_test

    def _load_labels(path):
        data = []
        for line in open(path, 'r'):
            data.append(int(line))
        return numpy.array(data, dtype=numpy.int32)

    def _load_images(path):
        data = []
        for line in open(path, 'r'):
            if norm:
                arr = [float(x) / 255. for x in line.split()]
            else:
                arr = [int(x) for x in line.split()]
            assert(len(arr) == 28 * 28)

            if dim == 1:
                data.append(arr)
            elif dim == 2:
                arr2 = []
                for i in range(28):
                    arr2.append(arr[i * 28: (i + 1) * 28])
                data.append(arr2)
            else:
                raise Exception('not yet implemented')

        if norm:
            return numpy.array(data, dtype=numpy.float32)
        return numpy.array(data, dtype=numpy.int32)

    images_train = _load_images('{}/emnist-balanced-train-images-idx3-txt'.format(DATA_ROOT))
    labels_train = _load_labels('{}/emnist-balanced-train-labels-idx1-txt'.format(DATA_ROOT))
    images_test = _load_images('{}/emnist-balanced-test-images-idx3-txt'.format(DATA_ROOT))
    labels_test = _load_labels('{}/emnist-balanced-test-labels-idx1-txt'.format(DATA_ROOT))

    os.path.exists('.cache') or os.mkdir('.cache')
    numpy.savez(cache_path,
                images_train=images_train,
                labels_train=labels_train,
                images_test=images_test,
                labels_test=labels_test)

    return images_train, labels_train, images_test, labels_test
