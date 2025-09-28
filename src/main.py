import numpy, skfuzzy

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tabulate import tabulate

from sklearn.metrics import mean_squared_error, mean_absolute_error


START_X = 0
START_Y = 20
POINTS_NUMBER = 100
WINDOW_SIZE = (1400, 600)
INTERVALS_INPUTS_NUMBER = 6
INTERVALS_OUTPUTS_NUMBER = 9


def calculate_z(x, y):
    return numpy.sin(2 * numpy.sqrt(x * x + y * y)) / (
        numpy.sqrt(x * x + y * y) + 0.001
    )


def calculate_y(x):
    return numpy.sin(x) + numpy.cos(x / 2)


def get_plots(nrows, ncols) -> tuple[Figure, Axes, Axes]:
    window, axs = pyplot.subplots(nrows, ncols)
    if nrows == 1 and ncols == 1:
        return window, axs
    return window, *axs


def from_pixels_to_inches(size: tuple[int, int], window: Figure) -> tuple[float, float]:
    width, height = size
    return (width / window.dpi, height / window.dpi)


def display_distribution_functions(
    x, y, z, x_distributions, y_distributions, z_distributions
):
    window, xg_axes, yg_axes, zg_axes = get_plots(1, 3)
    window.set_size_inches(from_pixels_to_inches(WINDOW_SIZE, window))
    window.canvas.manager.set_window_title("Distribution functions")

    for x_distribution in x_distributions:
        xg_axes.plot(x, x_distribution)
    xg_axes.set_title("x distribution")

    for y_distribution in y_distributions:
        x_numbers = numpy.linspace(min(y), max(y), len(y_distribution))
        yg_axes.plot(x_numbers, y_distribution)
    yg_axes.set_title("y distribution")

    for z_distribution in z_distributions:
        x_numbers = numpy.linspace(min(z), max(z), len(z_distribution))
        zg_axes.plot(x_numbers, z_distribution)
    zg_axes.set_title("z distribution")


def print_values(x_means, y_means):
    print("Table of values")
    table = [["x/y"] + [str(x) for x in x_means]]
    for i in range(len(x_means)):
        x_mean, y_mean = x_means[i], y_means[i]
        z_value = calculate_z(x_mean, y_mean)
        row = [round(y_mean, 2)] + [z_value] * len(x_means)
        table.append(row)
    print(tabulate(table, tablefmt="grid"))


def get_best_functions_index_trimf(value, means, diff):
    best_value = float("-inf")
    index = -1
    for i, mean in enumerate(means):
        trimf = skfuzzy.trimf(numpy.array([value]), [mean - diff, mean, mean + diff])
        if trimf > best_value:
            best_value = trimf
            index = i
    return index


def get_best_functions_index_gaussian(value, means, sigma):
    best_value = float("-inf")
    index = -1
    for i, mean in enumerate(means):
        trimf = skfuzzy.gaussmf(numpy.array([value]), mean, sigma)
        if trimf > best_value:
            best_value = trimf
            index = i
    return index


def get_best_functions_index_trapmf(value, means, x_delta, k):
    best_value = float("-inf")
    index = -1
    for i, mean in enumerate(means):
        trimf = skfuzzy.trapmf(
            numpy.array([value]),
            [mean - x_delta, mean - x_delta / k, mean + x_delta / k, mean + x_delta],
        )
        if trimf > best_value:
            best_value = trimf
            index = i
    return index


def diagonal_model(x, y, z, x_means, y_means, z_means):
    window3, zo_axes = get_plots(1, 1)
    window3.set_size_inches(from_pixels_to_inches(WINDOW_SIZE, window3))
    window3.canvas.manager.set_window_title("Z output")

    x_sigma = (max(x) - min(x)) * 0.15
    y_sigma = (max(y) - min(y)) * 0.15
    z_sigma = (max(z) - min(z)) * 0.15

    x_gaussians = (skfuzzy.gaussmf(x, x_mean, x_sigma) for x_mean in x_means)
    y_gaussians = (
        skfuzzy.gaussmf(numpy.linspace(min(y), max(y), POINTS_NUMBER), y_mean, y_sigma)
        for y_mean in y_means
    )
    z_gaussians = (
        skfuzzy.gaussmf(numpy.linspace(min(z), max(z), POINTS_NUMBER), z_mean, z_sigma)
        for z_mean in z_means
    )

    display_distribution_functions(x, y, z, x_gaussians, y_gaussians, z_gaussians)

    print_values(x_means, y_means)

    print("Table of functions name")
    table = [["x/y"] + ["mx" + str(i + 1) for i in range(len(x_means))]]
    rules = {}
    for i in range(len(x_means)):
        x_mean, y_mean = x_means[i], y_means[i]
        z_value = calculate_z(x_mean, y_mean)
        best_functions_index = get_best_functions_index_gaussian(
            z_value, z_means, z_sigma
        )
        row = ["my" + str(i + 1)] + ["mf" + str(best_functions_index + 1)] * len(
            x_means
        )
        table.append(row)
        for j in range(len(x_means)):
            rules[(j, i)] = best_functions_index
    print(tabulate(table, tablefmt="grid"))

    for (x_number, y_number), z_number in rules.items():
        print(
            f"if (x is mx{x_number + 1}) and (y is my{y_number + 1}) then (z is mf{z_number + 1})"
        )

    z_output = []
    for x_value in x:
        y_value = calculate_y(x_value)
        best_x = get_best_functions_index_gaussian(x_value, x_means, x_sigma)
        best_y = get_best_functions_index_gaussian(y_value, y_means, y_sigma)
        best_z = rules[(best_x, best_y)]
        z_output.append(z_means[best_z])
    zo_axes.plot(x, z_output, label="Model")
    zo_axes.plot(x, z, label="True")
    zo_axes.legend()

    mse = mean_squared_error(z, z_output)
    mae = mean_absolute_error(z, z_output)
    print(f"Mean Squared Error {mse}")
    print(f"Mean Absolute Error {mae}")


def trapmf_model(x, y, z, x_means, y_means, z_means):
    window3, zo_axes = get_plots(1, 1)
    window3.set_size_inches(from_pixels_to_inches(WINDOW_SIZE, window3))
    window3.canvas.manager.set_window_title("Z output")

    x_delta = (max(x) - min(x)) * 0.15
    y_delta = (max(y) - min(y)) * 0.15
    z_delta = (max(z) - min(z)) * 0.15

    x_trumfs = (
        skfuzzy.trapmf(
            x,
            [
                x_mean - x_delta,
                x_mean - x_delta / 2,
                x_mean + x_delta / 2,
                x_mean + x_delta,
            ],
        )
        for x_mean in x_means
    )
    y_trumfs = (
        skfuzzy.trapmf(
            numpy.linspace(min(y), max(y), POINTS_NUMBER),
            [
                y_mean - y_delta,
                y_mean - y_delta / 2,
                y_mean + y_delta / 2,
                y_mean + y_delta,
            ],
        )
        for y_mean in y_means
    )
    z_trumfs = (
        skfuzzy.trapmf(
            numpy.linspace(min(z), max(z), POINTS_NUMBER),
            [
                z_mean - z_delta,
                z_mean - z_delta / 2,
                z_mean + z_delta / 2,
                z_mean + z_delta,
            ],
        )
        for z_mean in z_means
    )

    display_distribution_functions(x, y, z, x_trumfs, y_trumfs, z_trumfs)

    print_values(x_means, y_means)

    print("Table of functions name")
    table = [["x/y"] + ["mx" + str(i + 1) for i in range(len(x_means))]]
    rules = {}
    for y_index, y_mean in enumerate(y_means):
        row = ["my" + str(y_index + 1)]
        for x_index, x_mean in enumerate(x_means):
            z_value = calculate_z(x_mean, y_mean)
            best_functions_index = get_best_functions_index_trapmf(
                z_value, z_means, z_delta, 2
            )
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
        y_value = calculate_y(x_value)
        best_x = get_best_functions_index_trapmf(x_value, x_means, x_delta, 2)
        best_y = get_best_functions_index_trapmf(y_value, y_means, y_delta, 2)
        best_z = rules[(best_x, best_y)]
        z_output.append(z_means[best_z])
    zo_axes.plot(x, z_output, label="Model")
    zo_axes.plot(x, z, label="True")
    zo_axes.legend()

    mse = mean_squared_error(z, z_output)
    mae = mean_absolute_error(z, z_output)
    print(f"Mean Squared Error {mse}")
    print(f"Mean Absolute Error {mae}")


def gaussian_model(x, y, z, x_means, y_means, z_means):
    window3, zo_axes = get_plots(1, 1)
    window3.set_size_inches(from_pixels_to_inches(WINDOW_SIZE, window3))
    window3.canvas.manager.set_window_title("Z output")

    x_sigma = (max(x) - min(x)) * 0.15
    y_sigma = (max(y) - min(y)) * 0.15
    z_sigma = (max(z) - min(z)) * 0.15

    x_gaussians = (skfuzzy.gaussmf(x, x_mean, x_sigma) for x_mean in x_means)
    y_gaussians = (
        skfuzzy.gaussmf(numpy.linspace(min(y), max(y), POINTS_NUMBER), y_mean, y_sigma)
        for y_mean in y_means
    )
    z_gaussians = (
        skfuzzy.gaussmf(numpy.linspace(min(z), max(z), POINTS_NUMBER), z_mean, z_sigma)
        for z_mean in z_means
    )

    display_distribution_functions(x, y, z, x_gaussians, y_gaussians, z_gaussians)

    print_values(x_means, y_means)

    print("Table of functions name")
    table = [["x/y"] + ["mx" + str(i + 1) for i in range(len(x_means))]]
    rules = {}
    for y_index, y_mean in enumerate(y_means):
        row = ["my" + str(y_index + 1)]
        for x_index, x_mean in enumerate(x_means):
            z_value = calculate_z(x_mean, y_mean)
            best_functions_index = get_best_functions_index_gaussian(
                z_value, z_means, z_sigma
            )
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
        y_value = calculate_y(x_value)
        best_x = get_best_functions_index_gaussian(x_value, x_means, x_sigma)
        best_y = get_best_functions_index_gaussian(y_value, y_means, y_sigma)
        best_z = rules[(best_x, best_y)]
        z_output.append(z_means[best_z])
    zo_axes.plot(x, z_output, label="Model")
    zo_axes.plot(x, z, label="True")
    zo_axes.legend()

    mse = mean_squared_error(z, z_output)
    mae = mean_absolute_error(z, z_output)
    print(f"Mean Squared Error {mse}")
    print(f"Mean Absolute Error {mae}")


def trimf_model(x, y, z, x_means, y_means, z_means):
    window3, zo_axes = get_plots(1, 1)
    window3.set_size_inches(from_pixels_to_inches(WINDOW_SIZE, window3))
    window3.canvas.manager.set_window_title("Z output")

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

    display_distribution_functions(x, y, z, x_trimfs, y_trimfs, z_trimfs)

    print_values(x_means, y_means)

    print("Table of functions name")
    table = [["x/y"] + ["mx" + str(i + 1) for i in range(len(x_means))]]
    rules = {}
    for y_index, y_mean in enumerate(y_means):
        row = ["my" + str(y_index + 1)]
        for x_index, x_mean in enumerate(x_means):
            z_value = calculate_z(x_mean, y_mean)
            best_functions_index = get_best_functions_index_trimf(
                z_value, z_means, z_delta
            )
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
        y_value = calculate_y(x_value)
        best_x = get_best_functions_index_trimf(x_value, x_means, x_delta)
        best_y = get_best_functions_index_trimf(y_value, y_means, y_delta)
        best_z = rules[(best_x, best_y)]
        z_output.append(z_means[best_z])
    zo_axes.plot(x, z_output, label="Model")
    zo_axes.plot(x, z, label="True")
    zo_axes.legend()

    mse = mean_squared_error(z, z_output)
    mae = mean_absolute_error(z, z_output)
    print(f"Mean Squared Error {mse}")
    print(f"Mean Absolute Error {mae}")


def main():
    x = numpy.linspace(START_X, START_Y, POINTS_NUMBER)
    y = calculate_y(x)
    z = calculate_z(x, y)

    x_means = numpy.linspace(min(x), max(x), INTERVALS_INPUTS_NUMBER)
    y_means = numpy.linspace(min(y), max(y), INTERVALS_INPUTS_NUMBER)
    z_means = numpy.linspace(min(z), max(z), INTERVALS_OUTPUTS_NUMBER)

    window, xy_axes, xz_axes = get_plots(1, 2)
    window.set_size_inches(from_pixels_to_inches(WINDOW_SIZE, window))
    window.canvas.manager.set_window_title("Original functions")
    xy_axes.plot(x, y)
    xz_axes.plot(x, z)
    xy_axes.set_title("y")
    xz_axes.set_title("z")

    diagonal_model(x, y, z, x_means, y_means, z_means)

    pyplot.show()


if __name__ == "__main__":
    main()
