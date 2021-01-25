"""
    This script uses results produced by save_latent_encodings.py
    Separation in two skipts was simply needed to experiment with visual style without re-running the net every time
"""

from pathlib import Path
import pickle
import numpy as np

# visualization
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# ---- load data ---- 
all_encodings = np.load('tmp_enc.npy')
with open('tmp_data_folders.pkl', 'rb') as fp:
    classes = pickle.load(fp)

print(all_encodings.shape, len(classes))


# ----- Dim reduction -----
# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
# https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_tsne.html
tsne = TSNE(n_components=2, random_state=0)
enc_2d = tsne.fit_transform(all_encodings)

# ----- Visualize ------
# update class labeling
mapping = {
    'data_1000_tee_200527-14-50-42_regen_200612-16-56-43': 'Shirts and dresses',
    'data_1000_skirt_4_panels_200616-14-14-40': 'Skirts',
    'data_1000_pants_straight_sides_210105-10-49-02': 'Pants'
}
classes = np.array([mapping[label] for label in classes])

# define colors
colors = {
    'Shirts and dresses': (0.747, 0.236, 0.048), # (190, 60, 12)
    'Skirts': (0.048, 0.0290, 0.747),  # (12, 74, 190)
    'Pants': (0.025, 0.354, 0.152)  # (6, 90. 39)
}

# plot
plt.figure(figsize=(6, 5))

for label, color in colors.items():
    plt.scatter(enc_2d[classes == label, 0], enc_2d[classes == label, 1], color=color, label=label)
plt.legend()
plt.savefig('D:/MyDocs/GigaKorea/SIGGRAPH2021 submission materials/Latent space/tsne.pdf')
plt.show()