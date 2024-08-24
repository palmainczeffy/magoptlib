import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from magoptlib.visualization import create_fig


def test_create_fig():

    # Test data
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    z = [1, 2, 3, 4, 5]

    fig, ax = create_fig(x, y, z)

    assert isinstance(ax, Axes3D), "The plot should be a 3D plot"
    assert len(ax.collections) > 0, "No data points plotted"

    scatter = ax.collections[0]
    assert scatter.get_offsets().shape[0] == len(x), \
        "Incorrect number of points plotted"

    plt.close(fig)
