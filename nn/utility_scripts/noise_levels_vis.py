import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def draw_plot(x, y, ax, color='black'):
    ax.scatter(x, y, c=color)

    # trend line
    # https://www.tutorialspoint.com/how-can-i-draw-a-scatter-trend-line-using-matplotlib
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), c=color)


# https://wandb.ai/maria_korosteleva/Garments-Reconstruction/runs/25g4d6si/overview?workspace=user-maria_korosteleva
x = [0, 0.2, 0.5, 1]

seen_values = {
    'Panel L2': [1.5, 7.2, 11.4],
    '#Panels': [99.7, 34.5, 7.75],
    '#Edges': [99.7, 91.9, 80.1],
    'Rot L2': [0.04, 0.13, 0.2],
    'Transl L2': [3.22, 3.55, 5.85]
}
unseen_values = {
    'Panel L2': [1, 1, 2, 4],
    '#Panels': [1, 2, 3, 4],
    '#Edges': [1, 2, 3, 4],
    'Rot L2': [1, 2, 3, 4],
    'Transl L2': [1, 2, 3, 4]
}

# Appearance setup
# global plot edges color setup
# https://stackoverflow.com/questions/7778954/elegantly-changing-the-color-of-a-plot-frame-in-matplotlib/7944576
base_color = '#737373'

# https://matplotlib.org/stable/tutorials/introductory/customizing.html
matplotlib.rc(
    'axes', 
    titlesize='medium', linewidth=0.6)
    # edgecolor=base_color, titlecolor=base_color, labelcolor=base_color)

# Plot
figure, axis = plt.subplots(
    1, len(seen_values), 
    sharex=True, sharey=False,
    figsize=(3 * len(seen_values), 3))

for i, plot_name in enumerate(seen_values):

    # Draw values
    draw_plot(x, seen_values[plot_name], axis[i], 'green')
    draw_plot(x, unseen_values[plot_name], axis[i], 'orange')

    # Appearance
    axis[i].set_box_aspect(1)   # https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/axes_box_aspect.html
    axis[i].set_title(plot_name, y=0, pad=-25, verticalalignment="top")   # https://stackoverflow.com/questions/55239332/place-title-at-the-bottom-of-the-figure-of-an-axes
    #axis[i].tick_params(axis='x', colors=base_color)
    #axis[i].tick_params(axis='y', colors=base_color)
    axis[i].margins(0.2)  # https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.axes.Axes.margins.html

figure.tight_layout() # Automatically adjust layout spacing

plt.show()