import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_mAP = -np.inf
        self.best_epoch = 0
        self.epochs_since_improvement = 0

    def __call__(self, mAP, epoch):
        if mAP > self.best_mAP + self.delta:
            self.best_mAP = mAP
            self.best_epoch = epoch
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1

        if self.epochs_since_improvement >= self.patience:
            print(f"Early stopping triggered after {self.patience} epochs without improvement.")
            return True

        return False




