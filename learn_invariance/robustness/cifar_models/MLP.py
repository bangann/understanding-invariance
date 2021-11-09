import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # self.layer0 = nn.Sequential(
        #     nn.Linear(3*32*32, 512),
        #     nn.ReLU(),
        # )
        # self.layer1 = nn.Sequential(
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Linear(512, num_classes),
        # )
        self.layer0 = nn.Sequential(
            nn.Linear(3*32*32, 10000),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(10000,10000),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(10000, num_classes),
        )

        self.layers = [self.layer0, self.layer1, self.layer2]

    def forward(self, x: torch.Tensor, return_full_list=False, clip_grad=False, with_latent=False, fake_relu=False, no_relu=False):
        '''Forward pass

        Args:
            x: Input.
            return_full_list: Optional, returns all layer outputs.

        Returns:
            torch.Tensor or list of torch.Tensor.

        '''
        assert (not fake_relu) and (not no_relu), \
            "fake_relu and no_relu not yet supported for this architecture"

        def _clip_grad(v, min, max):
            v_tmp = v.expand_as(v)
            v_tmp.register_hook(lambda g: g.clamp(min, max))
            return v_tmp

        out = []
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        for layer in self.layers:
            x = layer(x)
            if clip_grad:
                x = _clip_grad(x, -clip_grad, clip_grad)
            out.append(x)

        if not return_full_list:
            out = out[-1]

        return out


def MLP3(**kwargs):
    return MLP(**kwargs)

mlp=MLP3

# def test():
#     net = MLP3()
#     y = net(torch.randn(1,3,32,32))
#     print(y.size())
