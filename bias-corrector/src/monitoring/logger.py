"""
Asynchronous prediction logger with queue-based buffering.
"""
import logging

import csv
from pathlib import Path
from datetime import datetime

from queue import Queue
from threading import Thread

from ..api.schemas import PredictionRequest, PredictionResponse


logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Asynchronous prediction logger that writes to CSV without blocking API.
    
    Uses a queue and background thread to batch-write predictions.
    """
    
    def __init__(self, log_dir: Path, buffer_size: int = 100):
        """
        Initialize prediction logger.
        
        Args:
            log_dir: Directory to store prediction logs.
            buffer_size: Number of predictions to buffer before flushing.
        """
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer_size = buffer_size
        self.queue: Queue = Queue()
        self.running = True
        
        # CSV file path (daily rotation)
        self.current_date = datetime.utcnow().date()
        self.log_file = self._get_log_file()
        
        # CSV fieldnames
        self.fieldnames = [
            "timestamp",
            "state",
            "district",
            "crop_key",
            "year",
            "latitude",
            "longitude",
            "wofost_yield",
            "corrected_yield",
            "correction_factor",
            "ics_ratio",
            "model_version",
            "ci_lower",
            "ci_upper",
            "warnings"
        ]
        
        # Initialize CSV file
        self._init_csv_file()
        
        # Start background worker
        self.worker_thread = Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
        logger.info(f"Prediction logger initialized (buffer_size={buffer_size})")
    
    def _get_log_file(self) -> Path:
        """Get current log file path."""
        date_str = self.current_date.strftime("%Y%m%d")
        return self.log_dir / f"predictions_{date_str}.csv"
    
    def _init_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            logger.info(f"Created log file: {self.log_file}")
    
    async def log_prediction(
        self,
        request: PredictionRequest,
        response: PredictionResponse
    ):
        """
        Log a prediction (non-blocking).
        
        Args:
            request: Original prediction request.
            response: Prediction response.
        """
        record = {
            "timestamp": response.timestamp.isoformat(),
            "state": request.state,
            "district": request.district,
            "crop_key": request.crop_key.value,
            "year": request.year,
            "latitude": request.latitude,
            "longitude": request.longitude,
            "wofost_yield": request.wofost_yield,
            "corrected_yield": response.corrected_yield,
            "correction_factor": response.correction_factor,
            "ics_ratio": response.ics_ratio,
            "model_version": response.model_version,
            "ci_lower": response.confidence_interval.lower if response.confidence_interval else None,
            "ci_upper": response.confidence_interval.upper if response.confidence_interval else None,
            "warnings": "|".join(response.warnings) if response.warnings else ""
        }
        
        # Add to queue (non-blocking)
        self.queue.put(record)
    
    def _worker(self):
        """Background worker that writes buffered predictions to CSV."""
        buffer = []
        
        while self.running:
            try:
                # Collect records from queue
                while len(buffer) < self.buffer_size:
                    try:
                        record = self.queue.get(timeout=1.0)
                        buffer.append(record)
                    except:
                        break
                
                # Write buffer if not empty
                if buffer:
                    # Check for date rollover
                    current_date = datetime.utcnow().date()
                    if current_date != self.current_date:
                        self.current_date = current_date
                        self.log_file = self._get_log_file()
                        self._init_csv_file()
                    
                    # Write to CSV
                    with open(self.log_file, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                        writer.writerows(buffer)
                    
                    logger.debug(f"Flushed {len(buffer)} predictions to {self.log_file}")
                    buffer.clear()
            
            except Exception as e:
                logger.error(f"Error in prediction logger worker: {e}")
    
    def shutdown(self):
        """Shutdown logger and flush remaining predictions."""
        logger.info("Shutting down prediction logger...")
        self.running = False
        
        # Wait for worker to finish
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        # Flush remaining queue
        buffer = []
        while not self.queue.empty():
            try:
                buffer.append(self.queue.get_nowait())
            except:
                break
        
        if buffer:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerows(buffer)
            logger.info(f"Flushed final {len(buffer)} predictions")
        
        logger.info("Prediction logger shutdown complete")
