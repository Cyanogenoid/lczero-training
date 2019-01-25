import torch
import torch.nn as nn
import torch.nn.init as init

import net


class Net(nn.Module):
    def __init__(self, residual_channels, residual_blocks, policy_channels, se_ratio):
        super().__init__()
        channels = residual_channels

        self.input_conv = ConvBlock(112, channels, 3, padding=1)

        blocks = [ResidualBlock(channels, se_ratio) for _ in range(residual_blocks)]
        self.residual_stack = nn.Sequential(*blocks)

        self.policy_head = PolicyHead(channels, policy_channels)
        self.value_head = ValueHead(channels, 32, 128)
        
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                init.xavier_normal_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            if isinstance(module, nn.BatchNorm2d):
                init.ones_(module.weight)
                init.zeros_(module.bias)

    def forward(self, x):
        x = self.input_conv(x)
        x = self.residual_stack(x)

        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def conv_and_linear_weights(self):
        return [m.weight for m in self.modules() if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)]

    def from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['net'])

    def export_proto(self, path):
        weights = [w.detach().cpu().numpy() for w in extract_weights(self)]
        proto = net.Net()
        proto.fill_net(weights, se=True)
        proto.save_proto(path)

    def import_proto(self, path):
        proto = net.Net()
        proto.parse_proto(path)
        weights = proto.get_weights()
        for model_weight, loaded_weight in zip(extract_weights(self), weights):
            model_weight[:] = torch.from_numpy(loaded_weight)

    def export_onnx(self, path):
        dummy_input = torch.randn(10, 112, 8, 8)
        input_names = ['input_planes']
        output_names = [ 'policy_output', 'value_output']
        torch.onnx.export(self, dummy_input, path, input_names=input_names, output_names=output_names, verbose=True)


class PolicyHead(nn.Module):
    def __init__(self, in_channels, policy_channels):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_channels, policy_channels, 1),
            Flatten(),
            nn.Linear(8 * 8 * policy_channels, 1858),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ValueHead(nn.Module):
    def __init__(self, in_channels, value_channels, lin_channels):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_channels, value_channels, 1),
            Flatten(),
            nn.Linear(value_channels * 8 * 8, lin_channels),
            nn.ReLU(inplace=True),
            nn.Linear(lin_channels, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.tanh()
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, se_ratio):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),

            SqueezeExcitation(channels, se_ratio),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_in = x

        x = self.layers(x)

        x = x + x_in
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, ratio):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, 2 * channels),
        )

    def forward(self, x):
        x_in = x

        n, c, h, w = x.size()
        x = x.view(n, c, -1).mean(dim=2)

        x = self.layers(x)
        x = x.view(n, 2 * c, 1, 1)
        scale, shift = x.chunk(2, dim=1)

        x = scale.sigmoid() * x_in + shift
        return x


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


'''
# TODO
class GhostBatchNorm2d(nn.Module):
    def __init__(self, channels, virtual_batch_size):
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # TODO should be able to do this with some reshaping here
        x = self.bn(x)
        return x
'''


def extract_weights(m):
    if isinstance(m, Net):
        yield from extract_weights(m.input_conv)
        for block in m.residual_stack:
            yield from extract_weights(block)
        yield from extract_weights(m.policy_head)
        yield from extract_weights(m.value_head)

    elif type(m) in {ConvBlock, PolicyHead, ValueHead, ResidualBlock, SqueezeExcitation}:
        yield from extract_weights(m.layers)

    elif isinstance(m, nn.Sequential):
        for layer in m:
            yield from extract_weights(layer)

    elif isinstance(m, nn.Conv2d):
        # PyTorch uses same weight layout as cuDNN, so no transposes needed
        yield m.weight
        # no convs with biases in this net, only here for completeness
        if m.bias is not None:
            yield m.bias

    elif isinstance(m, nn.BatchNorm2d):
        yield m.weight
        yield m.bias
        yield m.running_mean
        yield m.running_var

    elif isinstance(m, nn.Linear):
        # PyTorch uses same weight layout as cuDNN, so no transposes needed
        yield m.weight
        yield m.bias


if __name__ == '__main__':
    net = Net(256, 20, 80, 4)
    batch = torch.rand(1, 112, 8, 8)
    policy, value = net(batch)
