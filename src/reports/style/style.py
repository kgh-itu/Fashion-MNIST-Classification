import seaborn


def custom_plot_style(style="darkgrid"):
    style_dict = {'axes.edgecolor': '#444444',
                  "font.family": "monospace",
                  'xtick.color': '#444444',
                  'ytick.color': '#444444',
                  }

    seaborn.set_theme(style=style, rc=style_dict)
