import numpy, skfuzzy

from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from collections.abc import Iterable


START_X = 0
START_Y = 20
POINTS_NUMBER = 100
WINDOW_SIZE = (1400, 600)
INTERVALS_INPUTS_NUMBER = 6
INTERVALS_OUTPUTS_NUMBER = 9


def get_plots(nrows, ncols) -> tuple[Figure, Axes, Axes]:
    window, axs = pyplot.subplots(nrows, ncols)
    return window, *axs


def from_pixels_to_inches(size: tuple[int, int], window: Figure) -> tuple[float, float]:
    width, height = size
    return (width / window.dpi, height / window.dpi)


def get_original_values() -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    x = numpy.linspace(START_X, START_Y, POINTS_NUMBER)
    y = numpy.cos(x / 2) + numpy.sin(x / 3)
    z = 0.5 * numpy.sin(x + y) * numpy.cos(y)
    return x, y, z


def get_trimf(
    x: numpy.ndarray, y: numpy.ndarray, z: numpy.ndarray
) -> tuple[
    Iterable[numpy.ndarray], Iterable[numpy.ndarray], Iterable[numpy.ndarray]
]:

    x_means = numpy.linspace(min(x), max(x), INTERVALS_INPUTS_NUMBER)
    y_means = numpy.linspace(min(y), max(y), INTERVALS_INPUTS_NUMBER)
    z_means = numpy.linspace(min(z), max(z), INTERVALS_OUTPUTS_NUMBER)

    x_delta = (max(x) - min(x)) * 0.15
    y_delta = (max(y) - min(y)) * 0.15
    z_delta = (max(z) - min(z)) * 0.15

    x_trimf = (
        skfuzzy.trimf(x, [x_means[i] - x_delta, x_means[i], x_means[i] + x_delta])
        for i in range(len(x_means))
    )
    y_trimf = (
        skfuzzy.trimf(
            numpy.linspace(min(y), max(y), POINTS_NUMBER),
            [y_means[i] - y_delta, y_means[i], y_means[i] + y_delta],
        )
        for i in range(len(y_means))
    )
    z_trimf = (
        skfuzzy.trimf(
            numpy.linspace(min(z), max(z), POINTS_NUMBER),
            [z_means[i] - z_delta, z_means[i], z_means[i] + z_delta],
        )
        for i in range(len(z_means))
    )

    return x_trimf, y_trimf, z_trimf


def main():
    x, y, z = get_original_values()

    window, xy_axes, xz_axes = get_plots(1, 2)
    window.set_size_inches(from_pixels_to_inches(WINDOW_SIZE, window))
    window.canvas.manager.set_window_title("Original functions")
    xy_axes.plot(x, y)
    xz_axes.plot(x, z)
    xy_axes.set_title("y")
    xz_axes.set_title("z")

    x_trimfs, y_trimfs, z_trimfs = get_trimf(x, y, z)
    window, xt_axes, yt_axes, zt_axes = get_plots(1, 3)
    window.set_size_inches(from_pixels_to_inches(WINDOW_SIZE, window))
    window.canvas.manager.set_window_title("Trimf functions")

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

    pyplot.show()


if __name__ == "__main__":
    main()
