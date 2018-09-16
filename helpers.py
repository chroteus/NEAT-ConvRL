def size_after_conv(h,w, kernel_size, dilation=(1,1),stride=(1,1), padding=(0,0)):
    if type(kernel_size) == int:
        kernel_size = (kernel_size,kernel_size)
    if type(dilation) == int:
        dilation = (dilation,dilation)
    if type(stride) == int:
        stride = (stride,stride)
    if type(padding) == int:
        padding = (padding,padding)

    new_h = h + (2*padding[0]) - (dilation[0]*(kernel_size[0]-1)) - 1
    new_h /= stride[0]
    new_h = math.floor(new_h + 1)

    new_w = w + (2*padding[1]) - (dilation[1]*(kernel_size[1]-1)) - 1
    new_w /= stride[1]
    new_w = math.floor(new_w + 1)

    return (new_h,new_w)


def size_after_pool(h,w,  kernel_size, dilation=(1,1), stride=False, padding=(0,0)):
    if not stride: stride = kernel_size
    return size_after_conv(h,w, kernel_size, dilation,stride,padding)

def flat_size_after_conv(conv_module, h,w):
    last_outch = -1
    for m in conv_module:
        if m.__class__.__name__ == "Conv2d":
            h,w = size_after_conv(h,w, m.kernel_size, m.dilation, m.stride, m.padding)
            last_outch = m.out_channels
        elif m.__class__.__name__ == "MaxPool2d":
            h,w = size_after_pool(h,w, m.kernel_size, m.dilation, m.stride, m.padding)
    return h*w*last_outch
