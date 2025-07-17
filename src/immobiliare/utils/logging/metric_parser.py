# src/immobiliare/utils/metric_parser.py

import json
from typing import List, Dict
from pathlib import Path
from core_interfaces.metrics.imetricparser import IMetricParser


class JsonMetricParser(IMetricParser):

    def load_metrics(self, path: Path) -> List[Dict]:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def filter_by_method(self, metrics: List[Dict], method_name: str) -> List[Dict]:
        return [m for m in metrics if m.get("method") == method_name]
