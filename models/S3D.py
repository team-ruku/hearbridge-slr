import torch
from torch import nn

from .S3DBase import BasicConv3d, S3DBase

BLOCK2SIZE = {1: 64, 2: 192, 3: 480, 4: 832, 5: 1024}


class S3D(S3DBase):
    def __init__(self, in_channel=3, use_block=5, freeze_block=0, coord_conv=None):
        self.use_block = use_block
        super(S3D, self).__init__(
            in_channels=in_channel, use_block=use_block, coord_conv=coord_conv
        )
        self.freeze_block = freeze_block
        self.END_POINT2BLOCK = {
            0: "block1",
            3: "block2",
            6: "block3",
            12: "block4",
            15: "block5",
        }
        self.BLOCK2END_POINT = {blk: ep for ep, blk in self.END_POINT2BLOCK.items()}

        self.frozen_modules = []
        self.use_block = use_block

        if freeze_block > 0:
            for i in range(
                0, self.base_num_layers
            ):  # base  0,1,2,...,self.BLOCK2END_POINT[blk]
                module_name = "base.{}".format(i)
                submodule = self.base[i]
                assert submodule is not None, module_name
                if i <= self.BLOCK2END_POINT["block{}".format(freeze_block)]:
                    self.frozen_modules.append(submodule)

    def forward(self, x):
        x = self.base(x)
        return x


class S3DBackbone(torch.nn.Module):
    def __init__(
        self,
        in_channel=3,
        use_block=5,
        freeze_block=0,
        cfg_pyramid=None,
        coord_conv=None,
        use_shortcut=False,
    ):
        super(S3DBackbone, self).__init__()
        self.cfg_pyramid = cfg_pyramid
        self.backbone = S3D(
            in_channel=in_channel,
            use_block=use_block,
            freeze_block=freeze_block,
            coord_conv=coord_conv,
        )
        self.set_frozen_layers()
        self.out_features = BLOCK2SIZE[use_block]

        self.stage_idx = [0, 3, 6, 12, 15]
        self.stage_idx = self.stage_idx[:use_block]
        self.use_block = use_block

        self.use_shortcut = use_shortcut
        if use_shortcut:
            dims = [64, 192, 480, 832, 1024]
            k_sizes = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3)]
            strides = [(1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2)]
            paddings = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (1, 1, 1)]
            self.shortcut_lst = nn.ModuleList()
            for i in range(use_block - 1):
                self.shortcut_lst.append(
                    BasicConv3d(
                        dims[i], dims[i + 1], k_sizes[i], strides[i], paddings[i]
                    )
                )

        self.num_levels = 3

    def set_frozen_layers(self):
        for m in getattr(self.backbone, "frozen_modules", []):
            for param in m.parameters():
                # print(param)
                param.requires_grad = False
            m.eval()

    def forward(self, sgn_videos):
        (B, C, T_in, H, W) = sgn_videos.shape

        # feat3d = self.backbone(sgn_videos)
        fea_lst = []
        shortcut_fea_lst = []
        for i, layer in enumerate(self.backbone.base):
            sgn_videos = layer(sgn_videos)

            # shortcut
            if self.use_shortcut and i in self.stage_idx:
                if i in self.stage_idx[1:]:
                    sgn_videos = sgn_videos + self.shortcut_lst[
                        self.stage_idx.index(i) - 1
                    ](shortcut_fea_lst[-1])
                shortcut_fea_lst.append(sgn_videos)

            if i in self.stage_idx[self.use_block - self.num_levels :]:
                # first block is too shallow, drop it
                fea_lst.append(sgn_videos)
                # print(sgn_videos.shape)

        return {"sgn_feature": fea_lst[-1], "fea_lst": fea_lst}
