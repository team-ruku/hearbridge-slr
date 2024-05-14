from torch import nn


class LateralConn(nn.Module):
    def __init__(
        self,
        inchannels,
        outchannels,
        kernelSize=(7, 3, 3),
        ratio=(1, 2, 2),
        direction="rgb2pose",
        variant=None,
        interpolate=False,
        adaptFirst=False,
    ):
        super(LateralConn, self).__init__()
        # ratio: temporal, height, width
        # kernel_size: temporal, height, width. 7 for temporal, same as posec3d. 3 for spatial, intuitively set
        # if bidirection, then create two fuser
        assert direction in ["rgb2pose", "pose2rgb"]
        self.direction = direction
        self.variant = variant

        padding = (kernelSize[0] // 2, kernelSize[1] // 2, kernelSize[2] // 2)
        if self.direction == "rgb2pose":
            if interpolate:
                self.conv = nn.AvgPool3d(
                    kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0)
                )
            else:
                if adaptFirst:
                    self.conv = nn.Conv3d(
                        inchannels,
                        outchannels,
                        kernelSize,
                        ratio,
                        padding=0,
                        bias=False,
                    )
                else:
                    self.conv = nn.Conv3d(
                        inchannels, outchannels, kernelSize, ratio, padding, bias=False
                    )
        elif self.direction == "pose2rgb":
            if interpolate:
                if adaptFirst:
                    self.conv = nn.Upsample(size=(8, 7, 7), mode="trilinear")
                else:
                    self.conv = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")
            else:
                if adaptFirst:
                    self.conv = nn.ConvTranspose3d(
                        inchannels,
                        outchannels,
                        kernelSize,
                        stride=(1, 3, 3),
                        padding=padding,
                        output_padding=0,
                        bias=False,
                    )
                else:
                    if ratio == (1, 2, 2):
                        output_padding = (0, 1, 1)
                    elif ratio == (2, 1, 1):
                        output_padding = (1, 0, 0)
                    self.conv = nn.ConvTranspose3d(
                        inchannels,
                        outchannels,
                        kernelSize,
                        ratio,
                        padding,
                        output_padding,
                        bias=False,
                    )
        self.init_weights()

    def forward(self, x_rgb, x_pose):
        if self.direction == "rgb2pose":
            x_rgb = self.conv(x_rgb)
        elif self.direction == "pose2rgb":
            x_pose = self.conv(x_pose)

        return None, x_rgb + x_pose

    def init_weights(self):
        if isinstance(self.conv, nn.Conv3d) or isinstance(
            self.conv, nn.ConvTranspose3d
        ):
            nn.init.kaiming_normal_(self.conv.weight)
