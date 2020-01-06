import torch

class PolyKernelConv2d(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', cp=2, dp=3, learnable_cp=False, device='cuda:0'):
        super(PolyKernelConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, padding_mode)
        # cp is learnable; dp is a constant;
        self.cp = torch.nn.Parameter(torch.tensor(cp, requires_grad=learnable_cp)).to(device)
        self.dp = dp
    
    def compute_shape(self, x):
        # Calculate the output shape. Follow https://pytorch.org/docs/stable/nn.html#conv2d
        output_height = (x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        output_width = (x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return output_height, output_width

    def forward(self, x):
        # x: [batch_size, in_channels, height, width]; x_unfold: [batch_size, in_channels*kernel_size[0]*kernel_size[1], output_height*output_width] 
        x_unfold = torch.nn.functional.unfold(x, self.kernel_size, self.dilation, self.padding, self.stride)
        h, w = self.compute_shape(x)
        # weight: [out_channels, in_channels, kernel_size[0], kernel_size[1]
        weight = self.weight
        # mul: [batch_size, out_channels, output_height*output_width]
        mul = x_unfold.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
        
        if self.bias is not None:
            mul = mul.transpose(1,2)
            mul += self.bias
            mul = mul.transpose(1,2)

        # Polynomial kernel: see Equation (10) in paper 'Kervolutional Neural Networks'
        mul = (mul + self.cp)**self.dp
        # output: [batch_size, out_channels, output_height, output_width]
        output = mul.view(x.shape[0], -1, h, w)
        return output



