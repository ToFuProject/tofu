def test() :
    inv_direction=[0., 1., 2.]
    sign = [0, 0, 0]
    sign[0] = int(inv_direction[0] < 0)
    sign[1] = int(inv_direction[1] < 0)
    sign[2] = int(inv_direction[2] < 0)


if __name__ == '__main__':
    import timeit
    print(timeit.timeit("test()", setup="from __main__ import test", number=100000))
