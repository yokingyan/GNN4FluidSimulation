import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# from src.data.load_data_info import read_metadata
# from src.data.dataset import *
import pickle


def test():

    # info = read_metadata(r"test\test_dataset\2D")
    with open(os.path.join(base_dir, r'test\test_dataset\3D\test\0.pkl'), 'rb') as fp:
        data = pickle.load(fp)
        pt, p = data['particle_type'], data['position']
        print(pt.shape, p.shape)
        print(pt)
    # for key, values in info.items():
    #     print(key, values)


def defunc(func):
    def wrapper(*args, **kargs):
        print(f"Start!!! {wrapper.__name__}")
        f = func(*args, **kargs)
        return f
    return wrapper


@defunc
def func01(a, b):
    return a + b


if __name__ == '__main__':

    a = func01(1, 2)
    print(a)
