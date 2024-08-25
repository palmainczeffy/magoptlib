import matplotlib.pyplot as plt


def create_fig(x, y, z):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points
    ax.scatter(x, y, z, color='green', marker='o')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio for all axes
    ax.set_box_aspect([1, 1, 1])

    plt.title("Visualization of the measurement points")

    plt.show()

    return fig, ax
