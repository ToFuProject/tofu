def test() :
    inv_direction=[0., 1., 2.]
    sign = [1 if inv < 0 else 0 for inv in inv_direction]


if __name__ == '__main__':
    import timeit
    print(timeit.timeit("test()", setup="from __main__ import test", number=100000))

# import numpy as np

# def test():
#     vector = np.arange(3)+1.
#     inv = 1./vector


# if __name__ == '__main__':
#     import timeit
#     print(timeit.timeit("test()", setup="from __main__ import test"))
