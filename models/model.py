import torch

from .recognition import RecognitionNetwork


class SignLanguageModel(torch.nn.Module):
    def __init__(self, config, num, wordEmbTab=None) -> None:
        super().__init__()
        self.device = config["device"]
        self.recognition_network = RecognitionNetwork(
            config=config["model"]["RecognitionNetwork"],
            transformConfig=config["data"]["transform_cfg"],
            num=num,
            inputStreams=config["data"].get("input_streams", "rgb"),
            wordEmbTab=wordEmbTab,
        )

    def forward(self, sgn_videos, sgn_keypoints, **kwargs):
        model_outputs = self.recognition_network(sgn_videos, sgn_keypoints, **kwargs)
        # model_outputs['total_loss'] = model_outputs['recognition_loss']
        return model_outputs

    def predict_gloss_from_logits(self, gloss_logits, k=10):
        return self.recognition_network.decode(gloss_logits=gloss_logits, k=k)

    def set_eval(self):
        self.eval()


def buildModel(config, num, **kwargs):
    model = SignLanguageModel(config, num, **kwargs)
    return model.to(config["device"])
