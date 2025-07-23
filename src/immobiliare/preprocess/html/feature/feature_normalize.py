import math
from dataclasses import is_dataclass, asdict, replace
from typing import List, Dict, Union, Any

from immobiliare.core_interfaces.feature.ifeature_normalizer import IFeatureNormalizer
from immobiliare.utils.logging.logger_factory import LoggerFactory


def _to_dict(item: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    return asdict(item) if is_dataclass(item) else item


def _update_item(item, updates: Dict[str, float]):
    return replace(item, **updates) if is_dataclass(item) else {**item, **updates}


class FeatureNormalizer(IFeatureNormalizer):
    def __init__(self, features_to_normalize=None):
        self.features_to_normalize = features_to_normalize
        self.means = {}
        self.stds = {}
        self.logger = LoggerFactory.get_logger("normalizer")

    def fit(self, data: List[Union[Dict[str, Any], Any]]) -> None:
        if not data:
            self.logger.log_info("Nessun dato da normalizzare.")
            return

        sample = _to_dict(data[0])
        if self.features_to_normalize is None:
            self.features_to_normalize = [
                k for k, v in sample.items()
                if isinstance(v, (int, float)) and k not in ("position", "token")
            ]

        sums = {f: 0.0 for f in self.features_to_normalize}
        counts = {f: 0 for f in self.features_to_normalize}
        for item in data:
            item_dict = _to_dict(item)
            for f in self.features_to_normalize:
                val = item_dict.get(f)
                if val is not None:
                    sums[f] += val
                    counts[f] += 1

        for f in self.features_to_normalize:
            self.means[f] = sums[f] / counts[f] if counts[f] > 0 else 0.0

        sq_sums = {f: 0.0 for f in self.features_to_normalize}
        for item in data:
            item_dict = _to_dict(item)
            for f in self.features_to_normalize:
                val = item_dict.get(f)
                if val is not None:
                    diff = val - self.means[f]
                    sq_sums[f] += diff * diff

        for f in self.features_to_normalize:
            if counts[f] > 1:
                self.stds[f] = math.sqrt(sq_sums[f] / (counts[f] - 1))
            else:
                self.stds[f] = 0.0

    def transform(self, data: List[Union[Dict[str, Any], Any]]) -> List[Union[Dict[str, Any], Any]]:
        result = []
        for item in data:
            item_dict = _to_dict(item)
            updates = {}
            for f in self.features_to_normalize:
                val = item_dict.get(f)
                mean = self.means.get(f, 0.0)
                std = self.stds.get(f, 0.0)
                if val is not None:
                    updates[f] = (val - mean) / std if std > 0 else 0.0
            try:
                result.append(_update_item(item, updates))
            except Exception as e:
                self.logger.log_exception(f"Errore nel normalizzare {item}", e)
        return result

    def fit_transform(self, data: List[Union[Dict[str, Any], Any]]):
        self.fit(data)
        return self.transform(data)
