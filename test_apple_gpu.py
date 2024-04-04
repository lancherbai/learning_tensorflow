import os
from functools import wraps
import time
import keras


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


@timeit
def train():
    cifar = keras.datasets.cifar100
    (x_train, y_train), (x_test, y_test) = cifar.load_data()  # will cache data under ~/.keras/datasets/
    model = keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(32, 32, 3),
        classes=100, )

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
    # model.fit(x_train, y_train, epochs=5, batch_size=64)
    model.fit(x_train, y_train, epochs=1, batch_size=64)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Setting log levels
    train()
