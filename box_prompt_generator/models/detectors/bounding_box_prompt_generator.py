from typing import Dict, Optional, Tuple, Union
import torch
from torch import Tensor

from typing import Dict, List, Tuple, Union

from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector


@MODELS.register_module()
class BoundingBoxPromptGenerator(SingleStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

    def predict_by_feats(self,
                feats: list,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if self.with_neck:
            x = self.neck(x)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def test_step(self, data: Union[dict, tuple, list], return_feats: bool=False) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='predict', return_feats=return_feats)  # type: ignore

    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str, return_feats: bool=False) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, mode=mode, return_feats=return_feats)
        elif isinstance(data, (list, tuple)):
            results = self(*data, mode=mode, return_feats=return_feats)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                return_feats: bool=False) -> ForwardResults:
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, return_feats=return_feats)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True,
                return_feats: bool=False) -> SampleList:
        if not return_feats:
            x = self.extract_feat(batch_inputs, return_feats=return_feats)
        else:
            x, feats = self.extract_feat(batch_inputs, return_feats=return_feats)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)

        if isinstance(results_list[0], torch.Tensor):
            return results_list

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        if not return_feats:
            return batch_data_samples
        else:
            return batch_data_samples, feats

    def extract_feat(self, batch_inputs: Tensor, return_feats: bool=False) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        if not return_feats:
            x = self.backbone(batch_inputs)
            if self.with_neck:
                x = self.neck(x)
            return x
        else:
            feats = self.backbone(batch_inputs, out_indices=(2, 3, 4, 5))
            if self.with_neck:
                x = self.neck(feats[:-1])
            return x, feats
            