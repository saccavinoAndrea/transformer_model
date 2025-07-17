from pathlib import Path

from config_loader import ConfigLoader
from utils.logging.metric_parser import JsonMetricParser

if __name__ == '__main__':

    config = ConfigLoader.load()
    metrics_dir = config.logger_out_dir + config.logger_out_file_name

    parser = JsonMetricParser()
    metrics = parser.load_metrics(Path(metrics_dir))
    slow_ops = parser.filter_by_method(metrics, "fetch")
    print(slow_ops)
