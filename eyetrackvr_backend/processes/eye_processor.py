from ..types import EyeData, Algorithms, TRACKING_FAILED
from ..config import AlgorithmConfig, TrackerConfig
from ..utils import WorkerProcess, BaseAlgorithm
from cv2.typing import MatLike
from queue import Queue, Full
from copy import deepcopy
import numpy as np
import queue
import cv2
from playsound import playsound


class EyeProcessor(WorkerProcess):
    def __init__(
        self,
        tracker_config: TrackerConfig,
        image_queue: Queue[MatLike],
        osc_queue: Queue[EyeData],
        frontend_queue: Queue[MatLike],
    ):
        super().__init__(name=f"Eye Processor {str(tracker_config.name)}", uuid=tracker_config.uuid)
        # Synced variables
        self.frontend_queue = frontend_queue
        self.image_queue = image_queue
        self.osc_queue = osc_queue
        # Unsynced variables
        self.algorithms: list[BaseAlgorithm] = []
        self.config: AlgorithmConfig = tracker_config.algorithm
        self.tracker_position = tracker_config.tracker_position
        self.tracker_config: TrackerConfig = tracker_config
        self.isCalibrating = False
        self.dataPoints: list[EyeData] = []
        self.targetNumDataPoints = 1000

    def startup(self) -> None:
        self.setup_algorithms()

    # Checks the config for a calibration signal
    def calibrationWatcher(self): # Passing self in every function. okay
        self.logger.error(f"Starting calibration")
        if self.tracker_config.calibrationData.min_x < -10:
            self.tracker_config.calibrationData.min_x = 0
            self.isCalibrating = True # Fucking everything in Python is lower case. Just not True for some reason
            self.dataPoints.clear(); # Optional semi-colon
            self.targetNumDataPoints = 500

    def calculateCalibration(self):
        x_arr = np.array([data.x for data in self.dataPoints], dtype=np.float32)
        y_arr = np.array([data.y for data in self.dataPoints], dtype=np.float32)
        blink_arr = np.array([data.blink for data in self.dataPoints], dtype=np.float32)
        self.tracker_config.calibrationData.max_x = np.percentile(x_arr, 90)
        self.tracker_config.calibrationData.min_x = np.percentile(x_arr, 10)
        self.tracker_config.calibrationData.max_y = np.percentile(y_arr, 90)
        self.tracker_config.calibrationData.min_y = np.percentile(y_arr, 10)
        self.tracker_config.calibrationData.max_blink = np.percentile(blink_arr, 70)
        self.tracker_config.calibrationData.min_blink = np.percentile(blink_arr, 30)
        self.isCalibrating = False;
        for i in range(10):
            self.logger.error(f"Calibrated!!!")

    def map_value(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def run(self) -> None:
        try:
            current_frame = self.image_queue.get(block=True, timeout=0.5)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        except queue.Empty:
            return
        except Exception:
            self.logger.exception("Failed to get image from queue")
            return

        frames = []
        result = None # Assume no frames 
        # TODO: add support for running one algorithm for blink detection and another for gaze tracking
        for algorithm in self.algorithms:
            result, frame = algorithm.run(deepcopy(current_frame), self.tracker_position)
            frames.append(frame)
            if result == TRACKING_FAILED:
                self.logger.debug(f"Algorithm {algorithm.get_name()} failed to find a result")
                continue
            break

        if result is None:
            return;





        if self.isCalibrating and len(self.dataPoints) < self.targetNumDataPoints:
            self.dataPoints.append(result)
            self.logger.debug(len(self.dataPoints))
        elif self.isCalibrating and len(self.dataPoints) > self.targetNumDataPoints:
            self.calculateCalibration();
        elif not self.isCalibrating and len(self.dataPoints) > 0: 
            # apply calibration
            result.x = self.map_value(result.x, 0, 1, self.tracker_config.calibrationData.min_x, self.tracker_config.calibrationData.max_x)
            result.y = self.map_value(result.y, 0, 1, self.tracker_config.calibrationData.min_y, self.tracker_config.calibrationData.max_y)
            result.blink = self.map_value(result.blink, 0, 1, self.tracker_config.calibrationData.min_blink, self.tracker_config.calibrationData.max_blink)








        try:
            # This is kinda bad, i would like to use a bitwise or but ahsf modifies the frame dimensions
            frame_shape = max(frames, key=lambda x: x.shape[0] * x.shape[1]).shape
            current_frame = np.zeros(frame_shape, dtype=np.uint8)
            frame_weight = min(1.0 / (len(frames)), 0.5)
            for frame in frames:
                if frame.shape != frame_shape:
                    frame = cv2.resize(frame, (frame_shape[1], frame_shape[0]))
                current_frame = cv2.addWeighted(current_frame, 1 - frame_weight, frame, frame_weight, 1)
            # make dark colors darker and light colors lighter
            current_frame = cv2.addWeighted(current_frame, 1.5, current_frame, 0, 0)
            self.osc_queue.put(result)
            self.frontend_queue.put(current_frame, block=False)
        except Full:
            pass
        self.window.imshow(self.process_name(), current_frame)

    def shutdown(self) -> None:
        pass

    def on_tracker_config_update(self, tracker_config: TrackerConfig) -> None:
        self.config = tracker_config.algorithm
        self.tracker_position = tracker_config.tracker_position
        self.tracker_config = tracker_config
        self.calibrationWatcher();
        self.setup_algorithms()

    def setup_algorithms(self) -> None:
        from ..algorithms import Blob, HSF, HSRAC, Leap, AHSF

        self.algorithms.clear()
        for algorithm in self.config.algorithm_order:
            match algorithm:
                case Algorithms.BLOB:
                    self.algorithms.append(Blob(self))
                case Algorithms.HSF:
                    self.algorithms.append(HSF(self))
                case Algorithms.HSRAC:
                    self.algorithms.append(HSRAC(self))
                # case Algorithms.RANSAC:
                #     self.algorithms.append(RANSAC(self))
                case Algorithms.LEAP:
                    self.algorithms.append(Leap(self))
                case Algorithms.AHSF:
                    self.algorithms.append(AHSF(self))
                case _:
                    self.logger.warning(f"Unknown algorithm: {algorithm}")
