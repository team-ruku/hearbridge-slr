import warnings

import cv2
import lintel
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.structures import DetDataSample
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown, init_model

# from mmdet.registry import VISUALIZERS

warnings.filterwarnings("ignore")

detector_config = (
    "./mmdetection/configs/faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py"
)
detector_ckpt = (
    "./assets/keypoint/faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth"
)

pose_config = "./mmpose/configs/wholebody_2d_keypoint/topdown_heatmap/coco-wholebody/td-hm_hrnet-w48_dark-8xb32-210e_coco-wholebody-384x288.py"
pose_ckpt = (
    "./assets/keypoint/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth"
)


def getSelectedIndexs(videoFrames, numFrames=64):
    pad = None
    if videoFrames >= numFrames:
        start = (videoFrames - numFrames) // 2
        selectedIndex = np.arange(start, start + numFrames)
    else:
        remain = numFrames - videoFrames
        selectedIndex = np.arange(0, videoFrames)
        padLeft = remain // 2
        padRight = remain - padLeft
        pad = (padLeft, padRight)

    return selectedIndex, pad


def loadFrameNumsTo4DArray(videoBytes, frameNums):
    decodedFrames, width, height = lintel.loadvid_frame_nums(
        videoBytes, frame_nums=frameNums
    )
    decodedFrames = np.frombuffer(decodedFrames, dtype=np.uint8)
    decodedFrames = np.reshape(decodedFrames, newshape=(-1, height, width, 3))
    return decodedFrames


def padArray(videoArrays, pad):
    padLeft, padRight = pad
    if padLeft > 0:
        pad_img = videoArrays[0]
        pad = np.tile(
            np.expand_dims(pad_img, axis=0),
            tuple([padLeft] + [1] * (len(videoArrays.shape) - 1)),
        )
        videoArrays = np.concatenate([pad, videoArrays], axis=0)
    if padRight > 0:
        pad_img = videoArrays[-1]
        pad = np.tile(
            np.expand_dims(pad_img, axis=0),
            tuple([padRight] + [1] * (len(videoArrays.shape) - 1)),
        )
        videoArrays = np.concatenate([videoArrays, pad], axis=0)
    return videoArrays


def loadVideo(videoPath):
    video = cv2.VideoCapture(videoPath)
    videoFrames = 0
    while True:
        ret, _ = video.read()
        if not ret:
            break
        videoFrames += 1
    print(videoFrames)

    selectedIndex, pad = getSelectedIndexs(videoFrames)
    print(selectedIndex, pad)

    with open(videoPath, "rb") as f:
        videoBytes = f.read()

    videoArrays = loadFrameNumsTo4DArray(videoBytes, selectedIndex)
    if pad is not None:
        videoArrays = padArray(videoArrays, pad)

    videoArrays = torch.tensor(videoArrays).float()  # T,H,W,C
    videoArrays = torch.permute(videoArrays, (0, 3, 1, 2))  # T,C,H,W

    return videoArrays


def detectionInference(model, frames):
    init_default_scope("mmdet")
    results = []
    for frame in frames:
        result = inference_detector(model, frame[0])
        assert isinstance(result, DetDataSample)

        instances = result.pred_instances
        instances = instances[instances.scores >= 0.75]  # type: ignore
        results.append(instances.bboxes.cpu().numpy())  # type: ignore
    return results


def poseInference(model, frames, detectionResults):
    init_default_scope("mmpose")
    results = np.zeros((len(frames), 133, 2), dtype=np.float32)
    for i, (frame, detectionResult) in enumerate(zip(frames, detectionResults)):
        result = inference_topdown(
            model, frame[0], bboxes=detectionResult, bbox_format="xyxy"
        )
        results[i] = result[0].pred_instances.keypoints  # type: ignore
    return results


def main():
    detectorModel = init_detector(
        config=detector_config,
        checkpoint=detector_ckpt,
        device="cuda:0",
    )
    poseModel = init_model(
        config=pose_config,
        checkpoint=pose_ckpt,
        device="cuda:0",
    )

    videoArrays = loadVideo("./videos/apple.mp4")
    frames = videoArrays.numpy().transpose(0, 2, 3, 1) * 255  # [T,H,W,C]
    frames = np.uint8(frames)
    assert frames.shape
    frames = np.split(frames, frames.shape[0], axis=0)

    detectionResults = detectionInference(detectorModel, frames)
    poseResults = poseInference(poseModel, frames, detectionResults)
    print(poseResults)


if __name__ == "__main__":
    main()
