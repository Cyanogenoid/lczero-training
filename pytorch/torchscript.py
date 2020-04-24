import sys

import torch
import torch.nn as nn
import model


if torch.__version__ == '1.4.0':
    print('warning, results may be slow')


TRACE = True
torch._C._jit_set_profiling_mode(True)


FILENAME = sys.argv[1]


class SoftmaxedValue(nn.Module):
    def __init__(self):
        super().__init__()
        global FILENAME
        self.net = model.Net(128, 10, 128, 8)
        self.net.from_checkpoint(FILENAME)
#        self.net = model.Net(320, 24, 320, 10)
#        self.net.import_proto("63198.pb.gz")
        self.softmax = nn.Softmax(dim=1)
        self.net.conv_block[0].weight.data[:, 109, :, :] /= 99  # scale rule50 weights due to legacy reasons

    def forward(self, x):
        policy, value_z, value_q = self.net(x)
        value_z = self.softmax(value_z)
        value_q = self.softmax(value_q)
        return policy, value_z, value_q

#    def forward(self, x):
#        policy, value = self.net(x)
#        value = self.softmax(value)
#        return policy, value


net = SoftmaxedValue().eval().cuda()
dummy_input = torch.rand(32, 112, 8, 8).cuda()
net(dummy_input)
#torch.backends.cudnn.benchmark = False

if TRACE:
    dummy_input = torch.rand(32, 112, 8, 8).cuda()
    traced_model = torch.jit.trace(net, dummy_input)
else:
    traced_model = torch.jit.script(net)

output_filename = '-'.join(FILENAME.split('/')[-2:])
traced_model.save(f'model_{output_filename}')
