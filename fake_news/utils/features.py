import json
import logging
import os
import pickle
from copy import deepcopy
from functools import partial
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from fake_news.utils.constants import CANONICAL_SPEAKER_TITLES
from fake_news.utils.constants import CANONICAL_STATE
from fake_news.utils.constants import PARTY_AFFILIATIONS
from fake_news.utils.constants import SIX_WAY_LABEL_TO_BINARY

logging.basicConfig(
    format="%(levelname)s - %(asctime)s - %(filename)s - %(message)s",
    level=logging.DEBUG
)
LOGGER = logging.getLogger(__name__)


class Datapoint(Basemodel):
    id: Optional[str]
    statement_json: Optional[str]
    label: Optional[bool]
    statement: str
    subject: Optional[str]
    speaker: Optional[str]
    speaker_title: Optional[str]
    state_info: Optional[str]
    party_affiliation: Optional[str]
    barely_true_count: float
    false_count: float
    half_true_count: float
    mostly_true_count: float
    pants_fire_count: float
    context: Optional[str]
    justification: Optional[str]

def normalize_labels(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        normalized_datapoint = deepcopy(datapoint)
        # .lower().strip() should be outside of the bracket.
        normalized_datapoint["label"] = SIX_WAY_LABEL_TO_BINARY[datapoint["label"].lower().strip()]
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints

def normalize_and_clean_counts(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for idx, datapoint in datapoints:
        normalized_datapoint = deepcopy(datapoint)
        for count_col in ["barely_true_count",
                          "false_count",
                          "half_true_count",
                          "mostly_true_count",
                          "pants_fire_count"]:
            if count_col in normalized_datapoint:
                normalized_datapoint[count_col] = float(normalized_datapoint[count_col])
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints
            

def normalize_and_clean_state_info(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        normalized_datapoint = deepcopy(datapoint)
        # combined into one line
        old_state_info = normalized_datapoint["state_info"].lower().strip().replace("-", " ")
        if old_state_info in CANONICAL_STATE:
            old_state_info = CANONICAL_STATE[old_state_info]
        normalized_datapoint["state_info"] = old_state_info
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean_party_affiliations(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        normalized_datapoint = deepcopy(datapoint)
        # extra check maybe not needed
        normalized_datapoint["party_affiliation"] = datapoint["party_affiliation"].lower().strip()
        if normalized_datapoint["party_affiliation"] not in PARTY_AFFILIATIONS:
            normalized_datapoint["party_affiliation"] = "none"
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints


def normalize_and_clean(datapoints: List[Dict]) -> List[Dict]:
    return normalize_and_clean_speaker_title(
        normalize_and_clean_party_affiliations(
            normalize_and_clean_state_info(
                normalize_and_clean_counts(
                    normalize_labels(
                        datapoints
                    )
                )
            )
        )
    )
