from abc import ABC, abstractmethod

class DimensionalityReducer(ABC):
    def __init__(self, output_dim, random_state=42):
        self.output_dim = output_dim
        self.random_state = random_state

    @abstractmethod
    def fit(self, X):
        """
        Fits the reducer on X.
        """
        pass

    @abstractmethod
    def transform(self, X):
        """
        Embeds X using the fitted model and returns the embeddings.
        Returns:
            Y (np.ndarray)
        """
        pass

    def fit_transform(self, X):
        """
        Fits the reducer on X and returns the embeddings.
        """
        self.fit(X)
        return self.transform(X)
