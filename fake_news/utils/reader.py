import json
from typing import List

import sys
sys.path.append('/content')
sys.path.append('/content/fake_news')
from fake_news.utils.features import Datapoint

def read_json_data(datapath: str) -> List[Datapoint]:
    with open(datapath) as f:
        datapoints = json.load(f)
        json_data = [Datapoint(**point) for point in datapoints]
    return json_data