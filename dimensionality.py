import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from dataloader import imageDataset

class dimensionality_reduction:
    def __init__(self, path):
        self.path = path
        self.data = imageDataset(path)
        self.images = np.array([img for img, _ in self.data])
        self.images_flat = self.images.reshape(len(self.images), -1)

    # data = imageDataset("Training_data.txt")
    # images = np.array([img for img, _ in data])
    # images_flattened = images.reshape(len(images), -1)

    # tsne = TSNE(n_components=2, random_state=42)
    # tsne_results = tsne.fit_transform(images_flattened)

    # mds = MDS(n_components=2, random_state=42)
    # mds_results = mds.fit_transform(images_flattened)

    # plt.figure(figsize=(10, 8))
    # plt.scatter(mds_results[:, 0], mds_results[:, 1], c=np.argmax(data.labels, axis=1), cmap='tab10')
    # plt.colorbar()
    # plt.title('mds results')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    # plt.show()