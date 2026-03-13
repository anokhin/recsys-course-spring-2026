import logging
from dataclasses import dataclass, asdict
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from queue import SimpleQueue

from pythonjsonlogger import jsonlogger

from botify.experiment import Experiments


@dataclass
class Datum:
    timestamp: int
    user: int
    track: int
    time: float
    latency: float
    recommendation: int = None


class DataLogger:
    """
    Write the provided Datum to the local log file
    in json format.

    Use an object of this class to write logs of
    user events. These logs are subsequently loaded
    to HDFS for analysis.
    """

    def __init__(self, app):
        self.logger = logging.getLogger("data")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        handler = RotatingFileHandler(
            app.config["DATA_LOG_FILE"],
            maxBytes=app.config["DATA_LOG_FILE_MAX_BYTES"],
            backupCount=app.config["DATA_LOG_FILE_BACKUP_COPIES"],
        )
        formatter = jsonlogger.JsonFormatter()
        handler.setFormatter(formatter)

        self.log_queue = SimpleQueue()
        self.listener = QueueListener(self.log_queue, handler)
        self.listener.start()
        self.logger.handlers.clear()
        self.logger.addHandler(QueueHandler(self.log_queue))

        self.experiment_context = Experiments()

    def log(self, location, datum: Datum, experiments=None):
        values = asdict(datum)
        if experiments is None:
            values["experiments"] = {
                experiment.name: experiment.assign(datum.user).name
                for experiment in self.experiment_context.experiments
            }
        else:
            values["experiments"] = experiments
        self.logger.info(location, extra=values)

    def close(self):
        self.listener.stop()
