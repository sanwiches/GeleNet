"""Additive experimental wrapper; archived fixed-level model remains untouched."""
import torch
from torch.nn import functional as F
from model.GeleNet_models import GeleNet, KTM
from model.scale_prior import ScalePrior


class ScaleKTM(KTM):
    def __init__(self, channel=32, mode='adaptive'):
        super().__init__(channel)
        self.scale_prior = ScalePrior(channel, mode=mode)

    def forward(self, x2, x3, region_ids=None): # V
        x_sum = x2 + x3 # Q
        x_mul = x2 * x3 # K

        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x_sum.size()
        proj_query = self.query_conv(x_sum).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x_mul).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)

        if region_ids is None:
            raise ValueError("Multiscale experiment requires region_ids")
        bias = self.scale_prior(x_sum, region_ids)
        energy = energy + F.softplus(self.region_strength) * bias

        attention = self.softmax(energy)

        proj_value_2 = self.value_conv_2(x2).view(m_batchsize, -1, width * height)
        proj_value_3 = self.value_conv_3(x3).view(m_batchsize, -1, width * height)

        out_2 = torch.bmm(proj_value_2, attention.permute(0, 2, 1))
        out_2 = out_2.view(m_batchsize, C, height, width)
        out_2 = self.conv_2(self.gamma_2 * out_2 + x2)

        out_3 = torch.bmm(proj_value_3, attention.permute(0, 2, 1))
        out_3 = out_3.view(m_batchsize, C, height, width)
        out_3 = self.conv_3(self.gamma_3 * out_3 + x3)

        x_out = self.conv_out(out_2 + out_3)

        return x_out


class ScaleGeleNet(GeleNet):
    def __init__(self, channel=32, mode='adaptive'):
        super().__init__(channel)
        base = self.KTM_23
        replacement = ScaleKTM(channel, mode)
        result = replacement.load_state_dict(base.state_dict(), strict=False)
        if result.unexpected_keys or any(not k.startswith('scale_prior.') for k in result.missing_keys):
            raise RuntimeError('Unexpected KTM initialization mismatch')
        self.KTM_23 = replacement
