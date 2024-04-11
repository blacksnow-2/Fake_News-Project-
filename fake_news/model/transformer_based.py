import os
from typing import List
from typing import Dict
from typing import Optional

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
# Save the model periodically by monitoring a quantity.
# After training finishes, use best_model_path to retrieve the path
# to the best checkpoint file and best_model_score to retrieve its score.
# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification

from fake_news.model.base import Model
from fake_news.utils.dataloaders import FakeNewsTorchDataset
from fake_news.utils.features import Datapoint