import logging
import pickle
import os
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Precision = TP/ (TP+FP). Recall = TP(TP+FN). f1 score is harmonic mean of Precision and recall
# f1 = 2 * (1 / (1/Precision + 1/Recall))
from sklearn.metrics import f1_score
# Below for roc_auc_score
# https://www.evidentlyai.com/classification-metrics/explain-roc-curve
from sklearn.metrics import roc_auc_score

from model.base import Model

import sys
sys.path.append('/content')
sys.path.append('/content/fake_news')
from fake_news.utils.features import Datapoint
from fake_news.utils.features import TreeFeaturizer

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)

class RandomForestModel(Model):
    def __init__(self, config: Optional[Dict] = None):
        self.config = config
        model_cache_path = os.path.join(config["model_output_path"], "model.pkl")
        self.featurizer = TreeFeaturizer(os.path.join(config["featurizer_output_path"], "featurizer.pkl"), config)
        if "evaluate" in config and config["evaluate"] and not os.path.exists(model_cache_path):
            raise ValueError("Model output path does not exist but in `evaluate` mode!")
        if model_cache_path and os.path.exists(model_cache_path):
            LOGGER.info("Loading model from cache...")
            with open(model_cache_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            LOGGER.info("Initializing model from scratch...")
            self.model = RandomForestClassifier(**self.config["params"])
    
    def train(self,
              train_datapoints: List[Datapoint],
              val_datapoints: List[Datapoint] = None,
              cache_featurizer: Optional[bool] = False) -> None:
        self.featurizer.fit(train_datapoints)
        if cache_featurizer:
            feature_names = self.featurizer.get_all_feature_names()
            with open(os.path.join(self.config["model_output_path"],
                                   "feature_names.pkl"), "wb") as f:
                pickle.dump(feature_names, f)
            self.featurizer.save(os.path.join(self.config["featurizer_output_path"],
                                              "featurizer.pkl"))
        train_labels = [datapoint.label for datapoint in train_datapoints]
        LOGGER.info("Featurizing data from scratch...")
        train_features = self.featurizer.featurize(train_datapoints)
        # New additions for applying the featurizer on val_datapoints...
        if val_datapoints is not None:
            self.config["evaluate_val"] = True
            self.val_featurizer = TreeFeaturizer(os.path.join(self.config["featurizer_output_path"], "val_featurizer.pkl"), self.config)
            self.val_featurizer.fit(val_datapoints)
            val_labels = [datapoint.label for datapoint in val_datapoints]
            val_features = self.featurizer.featurize(val_datapoints)
            val_feature_names = self.val_featurizer.get_all_feature_names()
            with open(os.path.join(self.config["model_output_path"],
                                   "val_feature_names.pkl"), "wb") as f:
                pickle.dump(val_feature_names, f)
            self.val_featurizer.save(os.path.join(self.config["featurizer_output_path"],
                                              "val_featurizer.pkl"))
            val_data = (val_features, val_labels)
            with open(os.path.join(os.path.dirname(self.config["val_data_path"]), "val_features.pkl"), "wb") as f:
                pickle.dump(val_data, f)
            self.config["evaluate_val"] = False
        self.model.fit(train_features, train_labels)
    
    def compute_metrics(self, eval_datapoints: List[Datapoint], split: Optional[str] = None) -> Dict:
        expected_labels = [datapoint.label for datapoint in eval_datapoints]
        predicted_proba = self.predict(eval_datapoints)
        predicted_labels = np.argmax(predicted_proba, axis=1)
        accuracy = accuracy_score(expected_labels, predicted_labels)
        f1 = f1_score(expected_labels, predicted_labels)
        # ??
        auc = roc_auc_score(expected_labels, predicted_proba[:, 1])
        conf_mat = confusion_matrix(expected_labels, predicted_labels)
        tn, fp, fn, tp = conf_mat.ravel()
        split_prefix = "" if split is None else split
        return {
            f"{split_prefix} f1": f1,
            f"{split_prefix} accuracy": accuracy,
            f"{split_prefix} auc": auc,
            f"{split_prefix} true negative": tn,
            f"{split_prefix} false negative": fn,
            f"{split_prefix} false positive": fp,
            f"{split_prefix} true positive": tp,
        }
    
    def predict(self, datapoints: List[Datapoint]) -> np.array:
        # what is features??
        features = self.featurizer.featurize(datapoints)
        return self.model.predict_proba(features)
    
    def get_params(self) -> Dict:
        return self.model.get_params()
    
    def save(self, model_cache_path: str) -> None:
        LOGGER.info("Saving model to disk...")
        with open(model_cache_path, "wb") as f:
            pickle.dump(self.model, f)
