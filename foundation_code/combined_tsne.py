import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def load_embeddings(embeddings_file):
    embeddings = np.load(embeddings_file)
    return embeddings

def tsne(embedding_values, labels, plot_save):
    tsne_model = TSNE(n_components=2, random_state=42)
    embedded_values = tsne_model.fit_transform(embedding_values)

    plt.figure(figsize=(10, 8))

    malignant_indices = np.where(labels == 1)[0]  # Indices of malignant samples
    benign_indices = np.where(labels == 0)[0]  # Indices of benign samples

    # Plot malignant samples in red
    plt.scatter(embedded_values[malignant_indices, 0], embedded_values[malignant_indices, 1], label='Malignant')

    # Plot benign samples in blue
    plt.scatter(embedded_values[benign_indices, 0], embedded_values[benign_indices, 1], label='Benign')
    plt.title('Combined t-SNE Plot of INBreast Embedding Values')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.savefig(plot_save)
    plt.show()

# Load embeddings and labels
embeddings_file1 = "/home/tajamul/scratch/DA/SFDA/Foundataion_Models/focalnet_crops/models/vitdino/aiims/results_inbreast/embeddings_save.npy"
embeddings_file2 = "/home/tajamul/scratch/DA/SFDA/Foundataion_Models/focalnet_crops/models/vitb16_imgnet/aiims/results_inbreast/embeddings_save.npy"
embeddings_file3 ="/home/tajamul/scratch/DA/SFDA/Foundataion_Models/focalnet_crops/models/vit16_clip/aiims/results_inbreast/embeddings_save.npy"




labels_file = "/home/tajamul/scratch/DA/SFDA/Foundataion_Models/focalnet_crops/models/vitdino/aiims/results_inbreast/logits_labels.npy"
embedding_values1 = np.load(embeddings_file1)
embedding_values2 = np.load(embeddings_file2)
embedding_values3 = np.load(embeddings_file3)
concatenated_embeddings = np.concatenate((embedding_values1, embedding_values2, embedding_values3), axis=1)

labels = np.load(labels_file)

# Plot t-SNE
plot_save = "/home/tajamul/scratch/DA/SFDA/Foundataion_Models/dino_full_img/models/tsne_plot.png"
tsne(concatenated_embeddings, labels, plot_save)