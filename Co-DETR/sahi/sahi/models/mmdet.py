# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.cv import get_bbox_from_bool_mask
from sahi.utils.import_utils import check_requirements
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
from mmdet.core import DatasetEnum

logger = logging.getLogger(__name__)

class MmdetDetectionModel(DetectionModel):
    def __init__(
        self,
        dataset: Optional[str] = None,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.00001,
        category_mapping: Optional[Dict] = None,
    ):
        super().__init__(
            dataset,
            model_path,
            model,
            config_path,
            device,
            confidence_threshold,
            category_mapping,
        )
    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        # create model
        if self.dataset == 'fisheye8k':
            model = init_detector(self.config_path, self.model_path, device=self.device, dataset=DatasetEnum.FISHEYE8K)
        elif self.dataset == 'fisheye8klvis':
            model = init_detector(self.config_path, self.model_path, device=self.device, dataset=DatasetEnum.FISHEYE8KLVIS)

        self.set_model(model)

    def set_model(self, model: Any):
        """
        Sets the underlying MMDetection model.
        Args:
            model: Any
                A MMDetection model
        """

        # set self.model
        self.model = model
        if self.dataset == 'fisheye8k':
            self.category_mapping = {'0': 'bus', '1': 'bike', '2':'car', '3':'pedestrian', '4':'truck'}
        elif self.dataset == 'fisheye8klvis':
            self.category_mapping = {str(ind): category_name for ind, category_name in enumerate(model.CLASSES)}

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """
        prediction_result = inference_detector(self.model, image)
        prediction = {"bboxes": [], "scores": [], "labels": []}
        for label, output in enumerate(prediction_result):
            for o in output:
                prediction['bboxes'].append(o[:4])
                prediction['scores'].append(o[4])
                prediction['labels'].append(label)
        
        self._original_predictions = prediction

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return 5


    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """

        try:
            from pycocotools import mask as mask_utils

            can_decode_rle = True
        except ImportError:
            can_decode_rle = False

        original_predictions = [self._original_predictions]
        category_mapping = self.category_mapping

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # parse boxes and masks from predictions
        object_prediction_list_per_image = []
        for image_ind, original_prediction in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]

            boxes = original_prediction["bboxes"]
            scores = original_prediction["scores"]
            labels = original_prediction["labels"]
            
            object_prediction_list = []

            n_detects = len(labels)
            # process predictions
            for i in range(n_detects):

                bbox = boxes[i]
                score = scores[i]
                category_id = labels[i]
                category_name = category_mapping[str(category_id)]

                # ignore low scored predictions
                if score < self.confidence_threshold:
                    continue

                bool_mask = None

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=bool_mask,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)
        self._object_prediction_list_per_image = object_prediction_list_per_image
