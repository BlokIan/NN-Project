import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from dataloader import imageDataset


class dimensionality_reduction:
    def __init__(self, path):
        self.data = self._load_data(path)
        self.images = self._extract_images(self.data)
        self.images_flat = self._flatten_images(self.images)
        self.tsne_results = None
        self.mds_results = None

    def _load_data(self, path):
        return imageDataset(path)

    def _extract_images(self, data):
        return np.array([img for img, _ in self.data])

    def _flatten_images(self, images):
        return images.reshape(len(images), -1)

    def apply_tsne(self):
        tsne = TSNE(n_components=2)
        self.tsne_results = tsne.fit_transform(self.images_flat)
    
    def apply_mds(self):
        mds = MDS(n_components=2)
        self.mds_results = mds.fit_transform(self.images_flat)

    def plot_tsne(self):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.tsne_results[:, 0], self.tsne_results[:, 1], c=np.argmax(self.data.labels, axis=1), cmap='tab10')
        plt.colorbar()
        plt.title('mds results')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()

    def plot_mds(self):
        plt.figure(figsize=(10, 8))
        plt.scatter(self.mds_results[:, 0], self.mds_results[:, 1], c=np.argmax(self.data.labels, axis=1), cmap='tab10')
        plt.colorbar()
        plt.title('mds results')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()