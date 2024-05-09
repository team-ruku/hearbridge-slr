import warnings
from os import path

import cv2
import lintel
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.structures import DetDataSample
from mmengine.registry import init_default_scope

from datasets.dataset import buildDataset
from funcs.misc import loadConfig, loadCustomized, moveToDevice
from mmpose.apis import inference_topdown, init_model
from models.model import buildModel

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

part2Index = {
    "pose": list(range(11)),
    "hand": list(range(91, 133)),
    "mouth": list(range(71, 91)),
    "face_others": list(range(23, 71)),
}
for key in ["mouth", "face_others", "hand"]:
    part2Index[key + "_half"] = part2Index[key][::2]
    part2Index[key + "_1_3"] = part2Index[key][::3]


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

    videoArrays = loadFrameNumsTo4DArray(videoBytes, selectedIndex)  # T,H,W,C
    if pad is not None:
        videoArrays = padArray(videoArrays, pad)

    videoArrays = torch.tensor(videoArrays).float()  # T,H,W,C
    videoArrays /= 255

    videos = []
    videos.append(videoArrays)
    videos = torch.stack(videos, dim=0).permute(0, 1, 4, 2, 3)  # 1,T,C,H,W

    return videos


def detectionInference(model, frames):
    init_default_scope("mmdet")
    detections = []
    for frame in frames:
        result = inference_detector(model, frame[0])
        assert isinstance(result, DetDataSample)

        instances = result.pred_instances
        instances = instances[instances.scores >= 0.75]  # type: ignore
        detections.append(instances.bboxes.cpu().numpy())  # type: ignore
    return detections


def poseInference(model, frames, detectionResults, filterKeys):
    init_default_scope("mmpose")
    results = np.zeros((len(frames), 133, 3), dtype=np.float32)
    for i, (frame, detectionResult) in enumerate(zip(frames, detectionResults)):
        result = inference_topdown(
            model, frame[0], bboxes=detectionResult, bbox_format="xyxy"
        )

        instances = result[0].pred_instances
        visibility = instances.keypoints_visible[0][:, np.newaxis]  # type: ignore
        combined = np.hstack((instances.keypoints[0], visibility))  # type: ignore
        combined = combined.reshape((133, 3))

        results[i] = combined

    filtered = []
    for key in sorted(filterKeys):
        selected = part2Index[key]
        filtered.append(results[:, selected])
    filtered = np.concatenate(filtered, axis=1)
    filtered = torch.from_numpy(filtered).float()  # T,N,3

    keypoints = []
    keypoints.append(filtered)
    keypoints = torch.stack(keypoints, dim=0)  # 1,T,N,3

    return keypoints


def main():
    config = loadConfig()
    config["device"] = "cuda:0"
    torch.cuda.set_device(config["device"])

    dataset = buildDataset(config["data"])
    vocab = dataset.vocab
    num = len(vocab)
    wordEmbTab = []
    if dataset.wordEmbTab:
        for w in vocab:
            wordEmbTab.append(torch.from_numpy(dataset.wordEmbTab[w]))
        wordEmbTab = torch.stack(wordEmbTab, dim=0).float().to(config["device"])
    del vocab
    del dataset

    model = buildModel(config, num, wordEmbTab=wordEmbTab)
    modelPath = path.join(
        "assets",
        str(config["data"]["num"]),
        "best.ckpt",
    )
    stateDict = torch.load(modelPath, map_location="cuda")
    loadCustomized(model, stateDict["model_state"], verbose=True)

    detectorModel = init_detector(
        config=detector_config,
        checkpoint=detector_ckpt,
        device=config["device"],
    )
    poseModel = init_model(
        config=pose_config,
        checkpoint=pose_ckpt,
        device=config["device"],
    )

    videoArrays = loadVideo("./videos/table.mp4")
    frames = videoArrays[0].numpy().transpose(0, 2, 3, 1) * 255  # [T,H,W,3]
    frames = np.uint8(frames)
    assert frames.shape
    frames = np.split(frames, frames.shape[0], axis=0)

    detectionResults = detectionInference(detectorModel, frames)
    poseResults = poseInference(
        poseModel, frames, detectionResults, config["data"]["use_keypoints"]
    )

    st = 64 // 4
    end = st + 64 // 2

    videos = []
    videos.append(videoArrays)
    videos.append(videos[-1][:, st:end, ...])

    keypoints = []
    keypoints.append(poseResults)
    keypoints.append(keypoints[-1][:, st:end, ...])

    batch = {
        "videos": videos,
        "keypoints": keypoints,
    }

    moveToDevice(batch, config["device"])

    model.eval()
    outputs = model(batch["videos"], batch["keypoints"])
    print(outputs)


if __name__ == "__main__":
    main()
