from abc import ABC, abstractmethod

class Classifier(ABC):
    def __init__(self, random_state=42):
        self.random_state = random_state

    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Fits the classifier on the training data.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predicts labels for X.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """
        Predicts probabilities for X.
        Should return an array of shape (n_samples,) representing probabilities for the positive class.
        """
        pass
