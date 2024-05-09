import math
from copy import deepcopy

import torch
import torchvision

from funcs.gaussian import gen_gaussian_hmap_op
from funcs.loss import LabelSmoothCE

from .four_stream import S3D_four_stream
from .Visualhead import SepConvVisualHead


class RecognitionNetwork(torch.nn.Module):
    def __init__(
        self,
        config,
        transformConfig,
        num=2000,
        inputStreams=["rgb"],
        wordEmbTab=None,
    ):
        super().__init__()
        self.config = config
        self.input_streams = inputStreams
        self.fuse_method = config.get("fuse_method", None)
        self.heatmap_cfg = config.get("heatmap_cfg", {})
        self.traj_hmap_cfg = config.get("traj_hmap_cfg", {})
        self.transform_cfg = transformConfig
        self.preprocess_chunksize = self.heatmap_cfg.get("preprocess_chunksize", 16)
        self.word_emb_tab = wordEmbTab

        config["pyramid"] = config.get(
            "pyramid", {"version": None, "rgb": None, "pose": None}
        )
        self.visual_backbone = self.visual_backbone_keypoint = (
            self.visual_backbone_twostream
        ) = None
        if len(inputStreams) == 4:
            self.visual_backbone_fourstream = S3D_four_stream(
                use_block=config["s3d"]["use_block"],
                freeze_block=(
                    config["s3d"]["freeze_block"],
                    config["keypoint_s3d"]["freeze_block"],
                ),
                pose_inchannels=config["keypoint_s3d"]["in_channel"],
                flag_lateral=(
                    config["lateral"].get("pose2rgb", True),
                    config["lateral"].get("rgb2pose", True),
                    config["lateral"].get("rgb_low2high", True),
                    config["lateral"].get("rgb_high2low", True),
                    config["lateral"].get("pose_low2high", True),
                    config["lateral"].get("pose_high2low", True),
                ),
                lateral_variant=(
                    config["lateral"].get("variant_pose2rgb", None),
                    config["lateral"].get("variant_rgb2pose", None),
                ),
                lateral_ksize=tuple(config["lateral"].get("kernel_size", (1, 3, 3))),
                lateral_ratio=tuple(config["lateral"].get("ratio", (1, 2, 2))),
                lateral_interpolate=config["lateral"].get("interpolate", False),
                cfg_pyramid=config["pyramid"],
                fusion_features=config["lateral"].get(
                    "fusion_features", ["c1", "c2", "c3"]
                ),
            )

        if "visual_head" in config:
            HeadCLS = SepConvVisualHead
            language_apply_to = config.get("language_apply_to", "rgb_keypoint_joint")
            if "rgb" in inputStreams:
                rgb_head_cfg = deepcopy(config["visual_head"])
                if "rgb" not in language_apply_to:
                    rgb_head_cfg["contras_setting"] = None
                self.visual_head = HeadCLS(
                    cls_num=num, word_emb_tab=wordEmbTab, **rgb_head_cfg
                )
            else:
                self.visual_head = None

            if "keypoint" in inputStreams:
                keypoint_head_cfg = deepcopy(config["visual_head"])
                if "keypoint" not in language_apply_to:
                    keypoint_head_cfg["contras_setting"] = None
                self.visual_head_keypoint = HeadCLS(
                    cls_num=num, word_emb_tab=wordEmbTab, **keypoint_head_cfg
                )
            else:
                self.visual_head_keypoint = None

            if len(inputStreams) == 4:
                self.visual_head = self.visual_head_keypoint = None
                self.visual_head_rgb_h = HeadCLS(
                    cls_num=num, word_emb_tab=wordEmbTab, **rgb_head_cfg
                )
                self.visual_head_rgb_l = HeadCLS(
                    cls_num=num, word_emb_tab=wordEmbTab, **rgb_head_cfg
                )
                self.visual_head_kp_h = HeadCLS(
                    cls_num=num, word_emb_tab=wordEmbTab, **keypoint_head_cfg
                )
                self.visual_head_kp_l = HeadCLS(
                    cls_num=num, word_emb_tab=wordEmbTab, **keypoint_head_cfg
                )
                self.head_dict = {
                    "rgb-h": self.visual_head_rgb_h,
                    "rgb-l": self.visual_head_rgb_l,
                    "kp-h": self.visual_head_kp_h,
                    "kp-l": self.visual_head_kp_l,
                    "fuse": None,
                    "fuse-h": None,
                    "fuse-l": None,
                    "fuse-x-rgb": None,
                    "fuse-x-kp": None,
                }

            if self.fuse_method is not None and "four" in self.fuse_method:
                assert len(inputStreams) == 4
                joint_head_cfg = deepcopy(config["visual_head"])
                if "joint" not in language_apply_to:
                    joint_head_cfg["contras_setting"] = None
                if "catall" in self.fuse_method or "type3" in self.fuse_method:
                    joint_head_cfg["input_size"] = (
                        4 * config["visual_head"]["input_size"]
                    )
                    self.visual_head_fuse = HeadCLS(
                        cls_num=num, word_emb_tab=wordEmbTab, **joint_head_cfg
                    )
                    self.head_dict["fuse"] = self.visual_head_fuse
                if "type" in self.fuse_method:
                    joint_head_cfg["input_size"] = (
                        2 * config["visual_head"]["input_size"]
                    )
                    self.visual_head_fuse_h = HeadCLS(
                        cls_num=num, word_emb_tab=wordEmbTab, **joint_head_cfg
                    )
                    self.visual_head_fuse_l = HeadCLS(
                        cls_num=num, word_emb_tab=wordEmbTab, **joint_head_cfg
                    )
                    self.head_dict["fuse-h"] = self.visual_head_fuse_h
                    self.head_dict["fuse-l"] = self.visual_head_fuse_l

        label_smooth = config.get("label_smooth", 0.0)
        if isinstance(label_smooth, float) and label_smooth > 0:
            self.recognition_loss_func = LabelSmoothCE(
                lb_smooth=label_smooth, reduction="mean"
            )
        elif isinstance(label_smooth, str) and "word_emb_sim" in label_smooth:
            temp, lb_smooth, norm_type = (
                float(label_smooth.split("_")[-1]),
                float(label_smooth.split("_")[-2]),
                label_smooth.split("_")[-3],
            )
            variant = "word_sim"
            self.recognition_loss_func = LabelSmoothCE(
                lb_smooth=lb_smooth,
                reduction="mean",
                word_emb_tab=wordEmbTab,
                norm_type=norm_type,
                temp=temp,
                variant=variant,
            )
        else:
            self.recognition_loss_func = torch.nn.CrossEntropyLoss(reduction="mean")

        self.contras_setting = config["visual_head"].get("contras_setting", None)
        self.contras_loss_weight = config.get("contras_loss_weight", 1.0)
        if self.contras_setting and "dual" in self.contras_setting:
            self.contras_loss_func = LabelSmoothCE(
                reduction="mean", variant=self.contras_setting
            )

    def generate_batch_heatmap(self, keypoints, heatmap_cfg):
        # self.sigma
        # keypoints B,T,N,3
        B, T, N, D = keypoints.shape
        keypoints = keypoints.reshape(-1, N, D)
        chunk_size = int(math.ceil((B * T) / self.preprocess_chunksize))
        chunks = torch.split(keypoints, chunk_size, dim=0)

        heatmaps = []
        for chunk in chunks:
            # print(chunk.shape)
            hm = gen_gaussian_hmap_op(
                coords=chunk, **heatmap_cfg
            )  # sigma, confidence, threshold) #B*T,N,H,W
            N, H, W = hm.shape[-3:]
            heatmaps.append(hm)

        heatmaps = torch.cat(heatmaps, dim=0)  # B*T, N, H, W
        return heatmaps.reshape(B, T, N, H, W)

    def apply_spatial_ops(self, x, spatial_ops_func):
        ndim = x.ndim
        if ndim > 4:
            B, T, C_, H, W = x.shape
            x = x.view(-1, C_, H, W)
        chunks = torch.split(x, self.preprocess_chunksize, dim=0)
        transformed_x = []
        for chunk in chunks:
            transformed_x.append(spatial_ops_func(chunk))
        _, C_, H_o, W_o = transformed_x[-1].shape
        transformed_x = torch.cat(transformed_x, dim=0)
        if ndim > 4:
            transformed_x = transformed_x.view(B, T, C_, H_o, W_o)
        return transformed_x

    def augment_preprocess_inputs(
        self,
        sgn_videos=None,
        sgn_heatmaps=None,
        sgn_videos_low=None,
        sgn_heatmaps_low=None,
    ):
        rgb_h, rgb_w = (
            self.transform_cfg.get("img_size", 224),
            self.transform_cfg.get("img_size", 224),
        )
        if sgn_heatmaps is not None:
            hm_h, hm_w = self.heatmap_cfg["input_size"], self.heatmap_cfg["input_size"]

        if sgn_videos is not None:
            spatial_ops = []
            spatial_ops.append(torchvision.transforms.Resize([rgb_h, rgb_w]))
            spatial_ops = torchvision.transforms.Compose(spatial_ops)
            sgn_videos = self.apply_spatial_ops(sgn_videos, spatial_ops)
            if sgn_videos_low is not None:
                sgn_videos_low = self.apply_spatial_ops(sgn_videos_low, spatial_ops)
        if sgn_heatmaps is not None:
            spatial_ops = []
            spatial_ops.append(torchvision.transforms.Resize([hm_h, hm_w]))
            spatial_ops = torchvision.transforms.Compose(spatial_ops)
            sgn_heatmaps = self.apply_spatial_ops(sgn_heatmaps, spatial_ops)
            if sgn_heatmaps_low is not None:
                sgn_heatmaps_low = self.apply_spatial_ops(sgn_heatmaps_low, spatial_ops)

        if sgn_videos is not None:
            # convert to BGR for S3D
            if (
                "r3d" not in self.config
                and "x3d" not in self.config
                and "vit" not in self.config
            ):
                sgn_videos = sgn_videos[:, :, [2, 1, 0], :, :]  # B T 3 H W
            sgn_videos = sgn_videos.float()
            sgn_videos = (sgn_videos - 0.5) / 0.5
            sgn_videos = sgn_videos.permute(0, 2, 1, 3, 4).float()  # B C T H W
        if sgn_videos_low is not None:
            # convert to BGR for S3D
            if (
                "r3d" not in self.config
                and "x3d" not in self.config
                and "vit" not in self.config
            ):
                sgn_videos_low = sgn_videos_low[:, :, [2, 1, 0], :, :]  # B T 3 H W
            sgn_videos_low = sgn_videos_low.float()
            sgn_videos_low = (sgn_videos_low - 0.5) / 0.5
            sgn_videos_low = sgn_videos_low.permute(0, 2, 1, 3, 4).float()  # B C T H W
        if sgn_heatmaps is not None:
            sgn_heatmaps = (sgn_heatmaps - 0.5) / 0.5
            if sgn_heatmaps.ndim > 4:
                sgn_heatmaps = sgn_heatmaps.permute(0, 2, 1, 3, 4).float()
            else:
                sgn_heatmaps = sgn_heatmaps.float()
        if sgn_heatmaps_low is not None:
            sgn_heatmaps_low = (sgn_heatmaps_low - 0.5) / 0.5
            if sgn_heatmaps_low.ndim > 4:
                sgn_heatmaps_low = sgn_heatmaps_low.permute(0, 2, 1, 3, 4).float()
            else:
                sgn_heatmaps_low = sgn_heatmaps_low.float()
        return sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low

    def forwardImpl(self, labels, sgn_videos=None, sgn_keypoints={}, **kwargs):
        s3d_outputs = {}
        # Preprocess (Move from data loader)
        with torch.no_grad():
            # 1. generate heatmaps
            if "keypoint" in self.input_streams:
                assert sgn_keypoints is not None
                sgn_heatmaps = self.generate_batch_heatmap(
                    sgn_keypoints, self.heatmap_cfg
                )  # B,T,N,H,W or B,N,H,W
            else:
                sgn_heatmaps = None

            sgn_videos_low = kwargs.pop("sgn_videos_low", None)
            sgn_keypoints_low = kwargs.pop("sgn_keypoints_low", None)
            sgn_heatmaps_low = None
            if len(self.input_streams) == 4:
                sgn_heatmaps_low = self.generate_batch_heatmap(
                    sgn_keypoints_low, self.heatmap_cfg
                )

            # 2. augmentation and permute(colorjitter, randomresizedcrop/centercrop+resize, normalize-0.5~0.5, channelswap for RGB)
            sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low = (
                self.augment_preprocess_inputs(
                    sgn_videos, sgn_heatmaps, sgn_videos_low, sgn_heatmaps_low
                )
            )

        if len(self.input_streams) == 4:
            s3d_outputs = self.visual_backbone_fourstream(
                sgn_videos, sgn_videos_low, sgn_heatmaps, sgn_heatmaps_low
            )
            print(s3d_outputs["kp-h"])

        if "four" in self.fuse_method:
            outputs = {}
            keys_of_int = ["gloss_logits", "word_fused_gloss_logits", "topk_idx"]
            for head_name, fea in s3d_outputs.items():
                head_ops = self.head_dict[head_name](x=fea, labels=labels)
                for k in keys_of_int:
                    outputs[head_name + "_" + k] = head_ops[k]

            # deal with fuse heads
            effect_head_lst = ["rgb-h", "rgb-l", "kp-h", "kp-l"]
            for head_name, head in self.head_dict.items():
                if head is None or "fuse" not in head_name:
                    continue
                effect_head_lst.append(head_name)
                if head_name == "fuse":
                    fused_fea = torch.cat(
                        [
                            s3d_outputs["rgb-h"],
                            s3d_outputs["rgb-l"],
                            s3d_outputs["kp-h"],
                            s3d_outputs["kp-l"],
                        ],
                        dim=-1,
                    )
                    head_ops = head(x=fused_fea, labels=labels)
                elif head_name == "fuse-h":
                    fused_fea = torch.cat(
                        [s3d_outputs["rgb-h"], s3d_outputs["kp-h"]], dim=-1
                    )
                    head_ops = head(x=fused_fea, labels=labels)
                elif head_name == "fuse-l":
                    fused_fea = torch.cat(
                        [s3d_outputs["rgb-l"], s3d_outputs["kp-l"]], dim=-1
                    )
                    head_ops = head(x=fused_fea, labels=labels)
                elif head_name == "fuse-x-rgb":
                    fused_fea = torch.cat(
                        [s3d_outputs["rgb-h"], s3d_outputs["rgb-l"]], dim=-1
                    )
                    head_ops = head(x=fused_fea, labels=labels)
                elif head_name == "fuse-x-kp":
                    fused_fea = torch.cat(
                        [s3d_outputs["kp-h"], s3d_outputs["kp-l"]], dim=-1
                    )
                    head_ops = head(x=fused_fea, labels=labels)
                for k in keys_of_int:
                    outputs[head_name + "_" + k] = head_ops[k]
            del head_ops

            # ensemble prob and logits
            for head_name, head in self.head_dict.items():
                if head is None:
                    continue
                outputs["ensemble_all_gloss_logits"] = outputs.get(
                    "ensemble_all_gloss_logits",
                    torch.zeros_like(outputs["rgb-h_gloss_logits"]),
                ) + outputs[head_name + "_gloss_logits"].softmax(dim=-1)
                if "fuse" in head_name:
                    outputs["ensemble_last_gloss_logits"] = outputs.get(
                        "ensemble_last_gloss_logits",
                        torch.zeros_like(outputs["rgb-h_gloss_logits"]),
                    ) + outputs[head_name + "_gloss_logits"].softmax(dim=-1)
            outputs["ensemble_all_gloss_logits"] = outputs[
                "ensemble_all_gloss_logits"
            ].log()
            outputs["ensemble_last_gloss_logits"] = outputs[
                "ensemble_last_gloss_logits"
            ].log()

        else:
            raise ValueError

        return outputs

    def forward(self, sgn_videos=[], sgn_keypoints=[], **kwargs):
        print(type(sgn_videos[0]), type(sgn_keypoints[0]))
        print(type(sgn_videos[1]), type(sgn_keypoints[1]))
        print(sgn_videos[0].shape, sgn_keypoints[0].shape)
        print(sgn_videos[1].shape, sgn_keypoints[1].shape)
        return self.forwardImpl(
            torch.tensor([]).long().to(sgn_videos[0].device),
            sgn_videos=sgn_videos[0],
            sgn_keypoints=sgn_keypoints[0],
            sgn_videos_low=sgn_videos[1],
            sgn_keypoints_low=sgn_keypoints[1],
            **kwargs,
        )
