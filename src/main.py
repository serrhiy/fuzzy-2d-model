import numpy, skfuzzy

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tabulate import tabulate
from math import sin, cos


START_X = 0
START_Y = 20
POINTS_NUMBER = 100
WINDOW_SIZE = (1400, 600)
INTERVALS_INPUTS_NUMBER = 6
INTERVALS_OUTPUTS_NUMBER = 9


def get_plots(nrows, ncols) -> tuple[Figure, Axes, Axes]:
    window, axs = pyplot.subplots(nrows, ncols)
    if nrows == 1 and ncols == 1:
        return window, axs
    return window, *axs


def from_pixels_to_inches(size: tuple[int, int], window: Figure) -> tuple[float, float]:
    width, height = size
    return (width / window.dpi, height / window.dpi)


def get_original_values() -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    x = numpy.linspace(START_X, START_Y, POINTS_NUMBER)
    y = numpy.cos(x / 2) + numpy.sin(x / 3)
    z = 0.5 * numpy.sin(x + y) * numpy.cos(y)
    return x, y, z


def get_best_functions_index(value, means, diff):
    best_value = float("-inf")
    index = -1
    for i, mean in enumerate(means):
        trimf = skfuzzy.trimf(numpy.array([value]), [mean - diff, mean, mean + diff])
        if trimf > best_value:
            best_value = trimf
            index = i
    return index


def MSE(real: numpy.ndarray, model: numpy.ndarray) -> float:
    return numpy.mean((real - model) ** 2)


def MAE(real: numpy.ndarray, model: numpy.ndarray) -> float:
    return abs(numpy.mean(real - model))


def main():
    x, y, z = get_original_values()

    window, xy_axes, xz_axes = get_plots(1, 2)
    window.set_size_inches(from_pixels_to_inches(WINDOW_SIZE, window))
    window.canvas.manager.set_window_title("Original functions")
    xy_axes.plot(x, y)
    xz_axes.plot(x, z)
    xy_axes.set_title("y")
    xz_axes.set_title("z")

    window2, xt_axes, yt_axes, zt_axes = get_plots(1, 3)
    window2.set_size_inches(from_pixels_to_inches(WINDOW_SIZE, window))
    window2.canvas.manager.set_window_title("Trimf functions")

    window3, zo_axes = get_plots(1, 1)
    window3.set_size_inches(from_pixels_to_inches(WINDOW_SIZE, window))
    window3.canvas.manager.set_window_title("Z output")

    x_means = numpy.linspace(min(x), max(x), INTERVALS_INPUTS_NUMBER)
    y_means = numpy.linspace(min(y), max(y), INTERVALS_INPUTS_NUMBER)
    z_means = numpy.linspace(min(z), max(z), INTERVALS_OUTPUTS_NUMBER)

    x_delta = (max(x) - min(x)) * 0.15
    y_delta = (max(y) - min(y)) * 0.15
    z_delta = (max(z) - min(z)) * 0.15

    x_trimfs = (
        skfuzzy.trimf(x, [x_mean - x_delta, x_mean, x_mean + x_delta])
        for x_mean in x_means
    )
    y_trimfs = (
        skfuzzy.trimf(
            numpy.linspace(min(y), max(y), POINTS_NUMBER),
            [y_mean - y_delta, y_mean, y_mean + y_delta],
        )
        for y_mean in y_means
    )
    z_trimfs = (
        skfuzzy.trimf(
            numpy.linspace(min(z), max(z), POINTS_NUMBER),
            [z_mean - z_delta, z_mean, z_mean + z_delta],
        )
        for z_mean in z_means
    )

    for x_trimf in x_trimfs:
        xt_axes.plot(x, x_trimf)
    xt_axes.set_title("x trimf")

    for y_trimf in y_trimfs:
        x_numbers = numpy.linspace(min(y), max(y), len(y_trimf))
        yt_axes.plot(x_numbers, y_trimf)
    yt_axes.set_title("y trimf")

    for z_trimf in z_trimfs:
        x_numbers = numpy.linspace(min(z), max(z), len(z_trimf))
        zt_axes.plot(x_numbers, z_trimf)
    zt_axes.set_title("z trimf")

    print("Table of values")
    table = [["x/y"] + [str(x) for x in x_means]]
    for y_mean in y_means:
        row = [round(y_mean, 2)]
        for x_mean in x_means:
            z_value = 0.5 * sin(x_mean + y_mean) * cos(y_mean)
            row.append(round(z_value, 2))
        table.append(row)
    print(tabulate(table, tablefmt="grid"))

    print("Table of functions name")
    table = [["x/y"] + ["mx" + str(i + 1) for i in range(len(x_means))]]
    rules = {}
    for y_index, y_mean in enumerate(y_means):
        row = ["my" + str(y_index + 1)]
        for x_index, x_mean in enumerate(x_means):
            z_value = 0.5 * sin(x_mean + y_mean) * cos(y_mean)
            best_functions_index = get_best_functions_index(z_value, z_means, z_delta)
            row.append("mf" + str(best_functions_index + 1))
            rules[(x_index, y_index)] = best_functions_index
        table.append(row)
    print(tabulate(table, tablefmt="grid"))

    for (x_number, y_number), z_number in rules.items():
        print(
            f"if (x is mx{x_number + 1}) and (y is my{y_number + 1}) then (z is mf{z_number + 1})"
        )

    z_output = []
    for x_value in x:
        y_value = cos(x_value / 2) + sin(x_value / 3)
        best_x = get_best_functions_index(x_value, x_means, x_delta)
        best_y = get_best_functions_index(y_value, y_means, y_delta)
        best_z = rules[(best_x, best_y)]
        z_output.append(z_means[best_z])
    zo_axes.plot(x, z_output, label="Model")
    zo_axes.plot(x, z, label="True")
    zo_axes.legend()

    mean_squared_error = MSE(y, z_output)
    mean_absolute_error = MAE(y, z_output)

    print(f"Mean Squared Error {mean_squared_error}")
    print(f"Mean Absolute Error {mean_absolute_error}")

    pyplot.show()


if __name__ == "__main__":
    main()
