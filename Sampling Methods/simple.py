# Name: Mohammad Mahdi Vahedi
# Student number: 99109314
import os
import statistics
from matplotlib import pyplot as plt
import numpy as np


def bernoulli(p):
    x = np.random.random()
    if x <= p:
        return 1
    return 0


def exponential(lam_bda):
    x = np.random.random()
    return -np.log((1 - x)) / lam_bda


def geometric(p):
    count = 0
    while True:
        x = bernoulli(p)
        count += 1
        if x:
            break
    return count


def gaussian(mio, var):
    theta = np.random.random() * 2 * np.pi
    R = exponential(0.5)
    standard = np.sqrt(R) * np.cos(theta)
    return standard * np.sqrt(var) + mio


def first_distribution():
    x = np.random.random()
    if x < 0.3:
        return gaussian(4, 2)
    elif 0.3 <= x < 0.6:
        return gaussian(3, 2)
    else:
        return exponential(0.01)


def second_distribution():
    x = np.random.random()
    if x < 0.2:
        return gaussian(0, 10)
    elif 0.2 <= x < 0.4:
        return gaussian(20, 15)
    elif 0.4 <= x < 0.7:
        return gaussian(-10, 8)
    else:
        return gaussian(50, 25)


def third_distribution():
    x = np.random.random()
    if x < 0.2:
        return geometric(0.1)
    elif 0.2 <= x < 0.4:
        return geometric(0.5)
    elif 0.4 <= x < 0.6:
        return geometric(0.3)
    else:
        return geometric(0.04)


def plot_sample(index, array, path):
    if index == 1:
        number = "First"
        plt.hist(array, bins=[x for x in range(-20, 100, 1)])
    elif index == 2:
        number = "Second"
        plt.hist(array, bins=[x for x in range(-30, 90, 1)])
    else:
        number = "Third"
        plt.hist(array, bins=[x for x in range(1, 80, 1)])
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(number + " Distribution Histogram with 1000 samples")
    new_path = os.path.join(path, "pdf" + str(index) + "_sample.png")
    plt.savefig(new_path)
    plt.show()


def plot_pdf(index, path):
    if index == 1:
        number = "First"
        x = [i for i in range(-20, 100, 1)]
        y = list(map(lambda x: first_pdf(x), x))
    elif index == 2:
        number = "Second"
        x = [i for i in range(-30, 90, 1)]
        y = list(map(lambda x: second_pdf(x), x))
    else:
        number = "Third"
        x = [i for i in range(1, 80, 1)]
        y = list(map(lambda x: third_pdf(x), x))

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title(number + " Distribution PDF")
    plt.plot(x, y)
    new_path = os.path.join(path, "pdf" + str(index) + ".png")
    plt.savefig(new_path)
    plt.show()


def plot_first_distribution(path):
    array = []
    for i in range(1000):
        array.append(first_distribution())
    plot_sample(1, array, path)
    return array


def plot_second_distribution(path):
    array = []
    for i in range(1000):
        array.append(second_distribution())
    plot_sample(2, array, path)
    return array


def plot_third_distribution(path):
    array = []
    for i in range(1000):
        array.append(third_distribution())
    plot_sample(3, array, path)
    return array


def gaussian_pdf(mio, var, x):
    return (1 / np.sqrt(var * 2 * np.pi)) * np.exp(-1 * (x - mio) ** 2 / (2 * var))


def exponential_pdf(lam_bda, x):
    if x < 0:
        return 0
    else:
        return lam_bda * np.exp(-1 * lam_bda * x)


def geometric_pdf(p, x):
    return (1 - p) ** (x - 1) * p


def first_pdf(x):
    return 0.3 * gaussian_pdf(4, 2, x) + \
           0.3 * gaussian_pdf(3, 2, x) + \
           0.4 * exponential_pdf(0.01, x)


def second_pdf(x):
    return 0.2 * gaussian_pdf(0, 10, x) + \
           0.2 * gaussian_pdf(20, 15, x) + \
           0.3 * gaussian_pdf(-10, 8, x) + \
           0.3 * gaussian_pdf(50, 25, x)


def third_pdf(x):
    return 0.2 * geometric_pdf(0.1, x) + \
           0.2 * geometric_pdf(0.5, x) + \
           0.2 * geometric_pdf(0.3, x) + \
           0.4 * geometric_pdf(0.04, x)


def make_log_file(samples, path):
    mean = [0, 0, 0]
    var = [0, 0, 0]
    message = ""
    for i in range(len(samples)):
        mean[i] = statistics.mean(samples[i])
        var[i] = statistics.variance(samples[i], mean[i])
        message = message + str(i + 1) + " " + \
                  str(round(mean[i], 4)) + " " + \
                  str(round(np.sqrt(var[i]), 4)) + "\n"
    new_path = os.path.join(path, "log.txt")
    f = open(new_path, "w")
    f.write(message)
    f.close()


# Make Directory
path = os.path.join(os.getcwd(), "part1")
try:
    os.mkdir(path)
except OSError as error:
    print("File Already Exist!")

# First
first_sample_array = plot_first_distribution(path)
plot_pdf(1, path)

# Second
second_sample_array = plot_second_distribution(path)
plot_pdf(2, path)

# Third
third_sample_array = plot_third_distribution(path)
plot_pdf(3, path)

# Log
make_log_file([first_sample_array, second_sample_array, third_sample_array], path)
