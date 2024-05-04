import torch

from modelling.recognition import RecognitionNetwork


class SignLanguageModel(torch.nn.Module):
    def __init__(self, cfg, cls_num, word_emb_tab=None) -> None:
        super().__init__()
        self.device = cfg["device"]
        model_cfg = cfg["model"]
        self.frozen_modules = []
        self.recognition_network = RecognitionNetwork(
            cfg=model_cfg["RecognitionNetwork"],
            cls_num=cls_num,
            transform_cfg=cfg["data"]["transform_cfg"],
            input_streams=cfg["data"].get("input_streams", "rgb"),
            word_emb_tab=word_emb_tab,
        )

        if self.recognition_network.visual_backbone is not None:
            self.frozen_modules.extend(
                self.recognition_network.visual_backbone.get_frozen_layers()
            )
        if self.recognition_network.visual_backbone_keypoint is not None:
            self.frozen_modules.extend(
                self.recognition_network.visual_backbone_keypoint.get_frozen_layers()
            )
        if model_cfg["RecognitionNetwork"].get("only_tune_new_layer", False):
            assert self.recognition_network.visual_backbone_twostream is not None
            self.frozen_modules.extend(
                [
                    self.recognition_network.visual_backbone_twostream.rgb_stream,
                    self.recognition_network.visual_backbone_twostream.pose_stream,
                    self.recognition_network.visual_head,
                    self.recognition_network.visual_head_keypoint,
                ]
            )
            for name, params in self.recognition_network.named_parameters():
                if "unified_logits_fc" not in name and "lateral" not in name.lower():
                    params.requires_grad = False

    def forward(self, is_train, labels, sgn_videos, sgn_keypoints, epoch, **kwargs):
        model_outputs = self.recognition_network(
            is_train, labels, sgn_videos, sgn_keypoints, epoch, **kwargs
        )
        # model_outputs['total_loss'] = model_outputs['recognition_loss']
        return model_outputs

    def predict_gloss_from_logits(self, gloss_logits, k=10):
        return self.recognition_network.decode(gloss_logits=gloss_logits, k=k)

    def set_train(self):
        self.train()
        for m in self.frozen_modules:
            m.eval()

    def set_eval(self):
        self.eval()


def build_model(cfg, cls_num, **kwargs):
    model = SignLanguageModel(cfg, cls_num, **kwargs)
    return model.to(cfg["device"])
