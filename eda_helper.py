# my color maps
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def get_custom_palette():
    """
    Set and return a seaborn color palette.

    This function sets the seaborn color palette to a predefined set of colors and
    returns a matplotlib ListedColormap object that can be used in plotting functions.

    Returns:
        ListedColormap: A colormap object generated from the seaborn color palette.
    """
    colors = [
        "#4060AF",
        "#FF5416",
        "#FDC82F",
        "#00B2A9",
        "#E7E6E6",
        "#93509E",
        "#00A9E0",
        "#CF0071",
    ]
    sns.set_palette(colors)
    return ListedColormap(sns.color_palette(colors))


def get_custom_colormap():
    """
    Return a custom matplotlib colormap.

    This function creates a custom colormap from a predefined set of colors.

    Returns:
        LinearSegmentedColormap: A colormap object that can be used in plotting functions.
    """
    colors = ["#22335C", "#00B2A9", "#FDC82F"]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)
