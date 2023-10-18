"""
We do not keep the cross-epoch memories while the feature prototypes are extracted in an online fashion
More details can be checked at https://github.com/Shathe/SemiSeg-Contrastive
Thanks the authors for providing such a model to achieve the class-level separation.
"""

import torch

class FeatureMemory:
    
    def __init__(self,  elements_per_class=32, n_classes=2):
        self.elements_per_class = elements_per_class
        self.memory = [None] * n_classes
        self.n_classes = n_classes

    def add_features_from_sample_learned(self, model, features, class_labels):
        """
        Updates the memory bank with some quality feature vectors per class
        Args:
            model: segmentation model containing the self-attention modules (contrastive_class_selectors)
            features: BxFxWxH feature maps containing the feature vectors for the contrastive (already applied the projection head)
            class_labels:   BxWxH  corresponding labels to the [features]
            batch_size: batch size

        Returns:

        """
        features = features.detach()
        class_labels = class_labels.detach().cpu().numpy()

        elements_per_class = self.elements_per_class

        # for each class, save [elements_per_class]
        for c in range(self.n_classes):
            mask_c = class_labels == c  # get mask for class c
            selector = model.__getattr__('contrastive_class_selector_' + str(c))  # get the self attention module for class c
            features_c = features[mask_c, :] # get features from class c
            if features_c.shape[0] > 0:
                if features_c.shape[0] > elements_per_class:
                    with torch.no_grad():
                        # get ranking scores
                        rank = selector(features_c)
                        rank = torch.sigmoid(rank)
                        # sort them
                        _, indices = torch.sort(rank[:, 0], dim=0)
                        indices = indices.cpu().numpy()
                        features_c = features_c.cpu().numpy()
                        # get features with highest rankings
                        features_c = features_c[indices, :]
                        new_features = features_c[:elements_per_class, :]
                else:
                    new_features = features_c.cpu().numpy()
                    
                self.memory[c] = new_features



