import random
import numpy as np
from Kohonen import Kohonen as kohonen


def creatData(d_size=1000, condition=1):
    data = np.empty((d_size, 2), dtype=object)
    random.seed(11)
    if condition == 1:
        for i in range(d_size):
            data[i, 0] = random.randint(0, 1000) / 1000
            data[i, 1] = random.randint(0, 1000) / 1000
    elif condition == 2:  # condition == 2, 80% to be in the bottom right corner - non uniform
        for i in range(d_size):
            flag = random.randint(0, 100)
            if flag < 80:
                data[i, 0] = i / 1000
                data[i, 1] = random.randint(0, i) / 1000
            else:
                data[i, 0] = i / 1000
                data[i, 1] = random.randint(i, 1000) / 1000
    elif condition == 3:  # condition == 3, 80% to be in the bottom left square - non uniform
        c = 0
        for i in range(int(d_size * 0.2)):
            data[i, 0] = random.randint(500, 1000) / 1000
            data[i, 1] = random.randint(500, 1000) / 1000
            c = i
        for j in range(c, c + int(d_size * 0.8) + 1):
            data[j, 0] = random.randint(0, 500) / 1000
            data[j, 1] = random.randint(0, 500) / 1000
    else:  # condition == 4, donut
        n = 0
        while n < d_size:
            x = random.uniform(-2, 2)
            y = random.uniform(-2, 2)
            if 1 <= x ** 2 + y ** 2 <= 2:
                data[n, 0] = x
                data[n, 1] = y
                n += 1

    return data.astype(np.float64)


def main():
    data1 = creatData(condition=1)

    # Q1.1
    kohonen(h=1, w=15, r=2, alpha_start=0.35).fit(data=data1, interval=1001, print_mode=500)

    # Q1.2
    kohonen(h=1, w=200, r=60, alpha_start=0.4).fit(data=data1, interval=1001, print_mode=500)

    # Q1.3.1
    data2 = creatData(condition=2, d_size=1000)
    kohonen(h=1, w=30, r=50, alpha_start=0.5).fit(data=data2, interval=1001, print_mode=100)

    # Q1.3.2
    data3 = creatData(condition=3, d_size=1000)
    kohonen(h=1, w=30, r=50, alpha_start=0.5).fit(data=data3, interval=1001, print_mode=100)

    # Q2
    data4 = creatData(condition=4, d_size=1000)
    kohonen(h=1, w=30, r=15, alpha_start=0.6).fit(data=data4, interval=1001, print_mode=100)


if __name__ == '__main__':
    main()
