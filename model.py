import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt



def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=1,
                              padding=padding)

    def forward(self, x):
        out = self.conv(x)
        return out


class B_T_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_c, in_c, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        y = self.conv1(x)
        return y



'''
Multi-level feature extraction for LR-HSI and HR-MSI
'''
class encoder_hs(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(encoder_hs, self).__init__()
        self.in_channels = in_channels

        self.conv_1x_1_spa = nn.Conv2d(in_channels=self.in_channels, out_channels=mid_channels * 4, kernel_size=3, padding=1)
        self.conv_1x_2_spa = nn.Conv2d(in_channels=mid_channels * 4, out_channels=mid_channels * 4, kernel_size=3, padding=1)


        self.conv_1x_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=mid_channels * 4, kernel_size=1, padding=0)
        self.conv_1x_2 = nn.Conv2d(in_channels=mid_channels * 4, out_channels=mid_channels * 4, kernel_size=1, padding=0)

        self.up_1x_2x = nn.ConvTranspose2d(in_channels=mid_channels * 4, out_channels=mid_channels * 4, kernel_size=4, stride=2, padding=1)
        self.conv_2x_1 = nn.Conv2d(in_channels=mid_channels * 4, out_channels=mid_channels * 2, kernel_size=1, padding=0)
        self.conv_2x_2 = nn.Conv2d(in_channels=mid_channels * 2, out_channels=mid_channels * 2, kernel_size=1, padding=0)

        self.up_2x_4x = nn.ConvTranspose2d(in_channels=mid_channels * 2, out_channels=mid_channels * 2, kernel_size=4, stride=2, padding=1)
        self.conv_4x_1 = nn.Conv2d(in_channels=mid_channels * 2, out_channels=mid_channels, kernel_size=1, padding=0)
        self.conv_4x_2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=1, padding=0)

        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):

        out1 = self.LeakyReLU(self.conv_1x_1(x))
        out1 = self.LeakyReLU(self.conv_1x_2(out1))
        out1_spa = self.LeakyReLU(self.conv_1x_1_spa(x))
        out1_spa = self.LeakyReLU(self.conv_1x_2_spa(out1_spa))

        out1_up = self.up_1x_2x(out1)
        out2 = self.LeakyReLU(self.conv_2x_1(out1_up))
        out2 = self.LeakyReLU(self.conv_2x_2(out2))

        out2_up = self.up_2x_4x(out2)
        out3 = self.LeakyReLU(self.conv_4x_1(out2_up))
        out3 = self.LeakyReLU(self.conv_4x_2(out3))

        return out1, out2, out3, out1_spa


class encoder_ms(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(encoder_ms, self).__init__()
        self.in_channels = in_channels

        self.conv_4x_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=mid_channels, kernel_size=3, padding=1)
        self.conv_4x_2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1)

        self.down_4x = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=4, padding=1, stride=2)
        self.conv_2x_1 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels * 2, kernel_size=3, padding=1)
        self.conv_2x_2 = nn.Conv2d(in_channels=mid_channels * 2, out_channels=mid_channels * 2, kernel_size=3, padding=1)

        self.down_2x = nn.Conv2d(in_channels=mid_channels * 2, out_channels=mid_channels * 2, kernel_size=4, padding=1, stride=2)
        self.conv_1x_1 = nn.Conv2d(in_channels=mid_channels * 2, out_channels=mid_channels * 4, kernel_size=3, padding=1)
        self.conv_1x_2 = nn.Conv2d(in_channels=mid_channels * 4, out_channels=mid_channels * 4, kernel_size=3, padding=1)

        self.LeakyReLU = nn.LeakyReLU()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out1 = self.LeakyReLU(self.conv_4x_1(x))
        out1 = self.LeakyReLU(self.conv_4x_2(out1))

        out1_mp = self.MaxPool(out1)
        out2 = self.LeakyReLU(self.conv_2x_1(out1_mp))
        out2 = self.LeakyReLU(self.conv_2x_2(out2))

        out2_mp = self.MaxPool(out2)
        out3 = self.LeakyReLU(self.conv_1x_1(out2_mp))
        out3 = self.LeakyReLU(self.conv_1x_2(out3))

        return out1, out2, out3

'''
Interactive Registration Module (IRM)
'''
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, v, k, q):
        # Compute attention
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output


class Cross_Attention_Module(nn.Module):
    def __init__(self,num_features):
        super(Cross_Attention_Module, self).__init__()
        self.num_features = num_features
        self.q = nn.Linear(self.num_features, self.num_features, bias=False)
        self.kv = nn.Linear(self.num_features, self.num_features * 2, bias=False)
        self.attention = ScaledDotProductAttention(temperature=self.num_features ** 0.5)

    def forward(self, x, x_s):
        b, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)

        q = self.q(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        kv = self.kv(x_s.permute(0, 2, 3, 1)).reshape(b, h, w, 2, c).permute(3, 0, 4, 1, 2)
        k, v = kv[0], kv[1]

        v_attn = self.attention(v, k, q)
        output = v_attn.view(b, c, h, w)

        return output


class Registration(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels,
                            padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(0.2),
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(Registration, self).__init__()

        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)

        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
            torch.nn.BatchNorm2d(mid_channel * 2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel * 2, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.LeakyReLU(0.2),
        )

        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, out_channel)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)

        bottleneck1 = self.bottleneck(encode_pool3)

        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        final_layer = self.final_layer(decode_block1)
        final_layer = final_layer.clamp(min=-5, max=5)
        return final_layer


# Generate deformation field
class Deformation_Field_Block(nn.Module):
    def __init__(self, num_features):
        super(Deformation_Field_Block, self).__init__()

        self.conv_f = conv3x3(in_channels=num_features * 2, out_channels=num_features)
        self.cross_blk = Cross_Attention_Module(num_features=num_features)
        self.conv_att = conv3x3(in_channels=num_features * 2, out_channels=num_features * 2)
        self.reg_field = Registration(in_channel=num_features * 2, out_channel=2)

    def forward(self, x_m, x_f):
        cross_up = self.cross_blk(x_f, x_m)
        cross_up = torch.cat((x_f, cross_up), dim=1)

        cross_down = self.cross_blk(x_m, x_f)
        cross_down = torch.cat((cross_down, x_m), dim=1)

        cross_att =  cross_up + cross_down
        flow = self.reg_field(cross_att)
        return flow


# Resampling
class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones([height, 1]),
                           torch.transpose(torch.unsqueeze(torch.linspace(0.0, width - 1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        if self.use_gpu == True:
            x_t = x_t.cuda(0)
            y_t = y_t.cuda(0)

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.cuda(0)
        return torch.squeeze(torch.reshape(x, (-1, 1)))

    def interpolate(self, im, x, y):
        im = F.pad(im, (0, 0, 1, 1, 1, 1, 0, 0))
        batch_size, height, width, channels = im.shape
        batch_size, out_height, out_width = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        x = x + 1
        y = y + 1

        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width * height
        base = self.repeat(torch.arange(0, batch_size) * dim1, out_height * out_width)

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1, 0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1, 0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1, 0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1, 0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1, 0).expand(dim, channels))

        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1, 0)
        wb = (dx * (1 - dy)).transpose(1, 0)
        wc = ((1 - dx) * dy).transpose(1, 0)
        wd = ((1 - dx) * (1 - dy)).transpose(1, 0)

        output = torch.sum(torch.squeeze(torch.stack([wa * Ia, wb * Ib, wc * Ic, wd * Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output

    def forward(self, deformation_matrix, moving_image):
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]

        batch_size, height, width = dx.shape

        x_mesh, y_mesh = self.meshgrid(height, width)

        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])
        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)


'''
Spectral Recalibration and Fusion Module(SEFU)
'''
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, drop):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.LeakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        x = self.linear(x)
        x = self.LeakyReLU(x)
        x = self.dropout(x)

        return x


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim, drop):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2, drop)

    def forward(self, x, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(x)
        out = gamma * out + beta

        return out

# spatial information fusion branch
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, xf, fm):
        max_pool_f, _ = torch.max(xf, dim=1, keepdim=True)
        loc_pool_f = torch.mean(xf, dim=1, keepdim=True)
        max_pool_m, _ = torch.max(fm, dim=1, keepdim=True)
        loc_pool_m = torch.mean(fm, dim=1, keepdim=True)
        xf_att = self.LeakyReLU(self.conv(torch.cat([max_pool_f, loc_pool_f], dim=1)))
        fm_att = self.LeakyReLU(self.conv(torch.cat([max_pool_m, loc_pool_m], dim=1)))
        out_xf = xf_att * xf
        out_fm = fm_att * fm
        out = out_xf + out_fm

        return out


class StyledGenerator(nn.Module):

    def __init__(self, code_dim, n_mlp, drop):
        super().__init__()
        layers = nn.ModuleList()
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim, drop))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)

    def forward(self, x):
        styles = []
        if type(x) not in (list, tuple):
            x = [x]

        for i in x:
            styles.append(self.style(i))

        return styles


class StyledConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, n_mlp, drop, kernel_size, padding):
        super(StyledConvBlock, self).__init__()

        self.style_g = StyledGenerator(style_dim, n_mlp, drop)
        self.SpatialAttn = SpatialAttention()
        self.conv1 = EqualConv2d(in_channel, out_channel, kernel_size, padding)

        self.lrelu1 = nn.LeakyReLU()
        self.AdaIN1 = AdaptiveInstanceNorm(out_channel, style_dim, drop)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding)

        self.lrelu2 = nn.LeakyReLU()
        self.AdaIN2 = AdaptiveInstanceNorm(out_channel, style_dim, drop)

    def forward(self, xf, fm, style):
        styles = self.style_g(style)
        styles = torch.stack(styles)
        styles = styles.squeeze(0)

        out = self.AdaIN1(xf, styles)
        Att1 = self.SpatialAttn(fm, out)
        out = out + Att1
        out = self.conv1(out)
        out = self.lrelu1(out)
        out = self.AdaIN2(out, styles)
        Att2 = self.SpatialAttn(fm, out)
        out = out + Att2
        out = self.conv2(out)
        out = self.lrelu2(out)
        out = out + fm
        return out

'''
Progressive Multi-Iteration Registration-Fusion Co-Optimization Network (PMI-RFCoNet)
'''
class SuperResolutionModel(nn.Module):
    def __init__(self,
                 drop,
                 in_channels,
                 out_channels,
                 mid_channels,
                 factor,
                 num_fc):
        super(SuperResolutionModel, self).__init__()

        # Setting
        self.drop = drop
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        self.num_fc = num_fc
        self.channel = mid_channels * 4

        # encoder
        self.encoder_LR = encoder_hs(in_channels=self.in_channels, mid_channels=mid_channels)
        self.encoder_HR = encoder_ms(in_channels=4, mid_channels=mid_channels)


        self.w_GAP_1x = nn.AdaptiveAvgPool2d((1, 1))
        self.w_GAP_2x = nn.AdaptiveAvgPool2d((1, 1))
        self.w_GAP_4x = nn.AdaptiveAvgPool2d((1, 1))

        self.w_flatten_1x = self.channel
        self.w_flatten_2x = int(self.channel / 2)
        self.w_flatten_4x = int(self.channel / 4)


        # Deformation Field
        self.field_1x = Deformation_Field_Block(num_features=self.channel)
        self.field_2x = Deformation_Field_Block(num_features=int(self.channel / 2))
        self.field_4x = Deformation_Field_Block(num_features=int(self.channel / 4))

        # STN
        self.STN_1x = SpatialTransformation()
        self.STN_2x = SpatialTransformation()
        self.STN_4x = SpatialTransformation()


        self.StyleBlock_1x = StyledConvBlock(in_channel=self.channel,
                                             out_channel=self.channel,
                                             style_dim=self.channel,
                                             n_mlp=self.num_fc,
                                             drop=self.drop,
                                             kernel_size=3,
                                             padding=1)

        self.StyleBlock_2x = StyledConvBlock(in_channel=int(self.channel / 2),
                                             out_channel=int(self.channel / 2),
                                             style_dim=int(self.channel / 2),
                                             n_mlp=self.num_fc,
                                             drop=self.drop,
                                             kernel_size=3,
                                             padding=1)

        self.StyleBlock_4x = StyledConvBlock(int(self.channel / 4),
                                             int(self.channel / 4),
                                             int(self.channel / 4),
                                             n_mlp=self.num_fc,
                                             drop=self.drop,
                                             kernel_size=3,
                                             padding=1)

        self.conv_2x = conv3x3(self.channel, int(self.channel / 2))
        self.conv_4x = conv3x3(int(self.channel / 2), int(self.channel / 4))
        self.to_hsi = conv3x3(int(self.channel / 4), self.in_channels)

        self.upscale_1x = B_T_Block(int(self.channel / 2))
        self.relu = nn.LeakyReLU(0.2)
        self.upscale_2x = B_T_Block(int(self.channel / 4))


    def forward(self, X_LR, X_HR):

        f_4x, f_2x, f_1x = self.encoder_HR(X_HR)
        lr_1x, lr_2x, lr_4x, lr_spa = self.encoder_LR(X_LR)

        w_1x = self.w_GAP_1x(lr_1x)
        w_2x = self.w_GAP_2x(lr_2x)
        w_4x = self.w_GAP_4x(lr_4x)

        # STAGE 1
        flow_1x = self.field_1x(lr_spa, f_1x).permute(0, 2, 3, 1)
        lr_1x = lr_1x.permute(0, 2, 3, 1)
        # Spatial Transform
        reg_1x = self.STN_1x(flow_1x, lr_1x).permute(0, 3, 1, 2)
        # Styletransfer
        b_w_1x = w_1x.size(0)
        w_1x = w_1x.view(b_w_1x, self.w_flatten_1x)
        LR_1x = self.StyleBlock_1x(reg_1x, f_1x, w_1x)
        # upscale
        reg_LR_2x = self.relu(self.conv_2x(LR_1x))
        reg_LR_2x = self.upscale_1x(reg_LR_2x)


        # STAGE 2
        flow_2x = self.field_2x(reg_LR_2x, f_2x).permute(0, 2, 3, 1)
        reg_LR_2x = reg_LR_2x.permute(0, 2, 3, 1)
        # spatial Transform
        reg_2x = self.STN_2x(flow_2x, reg_LR_2x).permute(0, 3, 1, 2)
        # StyleTransfer
        b_w_2x = w_2x.size(0)
        w_2x = w_2x.view(b_w_2x, self.w_flatten_2x)
        LR_2x = self.StyleBlock_2x(reg_2x, f_2x, w_2x)
        # upscale
        reg_LR_4x = self.relu(self.conv_4x(LR_2x))
        reg_LR_4x = self.upscale_2x(reg_LR_4x)


        # STAGE 3
        flow_4x = self.field_4x(reg_LR_4x, f_4x).permute(0, 2, 3, 1)
        reg_LR_4x = reg_LR_4x.permute(0, 2, 3, 1)
        # spatial Transform
        reg_4x = self.STN_4x(flow_4x, reg_LR_4x).permute(0, 3, 1, 2)
        # StyleTransfer
        b_w_4x = w_4x.size(0)
        w_4x = w_4x.view(b_w_4x, self.w_flatten_4x)
        LR_4x = self.StyleBlock_4x(reg_4x, f_4x, w_4x)

        output = self.relu(self.to_hsi(LR_4x))

        return output

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    model = SuperResolutionModel(drop=0,
                                 in_channels=102,
                                 out_channels=102,
                                 mid_channels=128,
                                 factor=4,
                                 num_fc=4).to(device)
    a = torch.rand(1, 102, 40, 40).to(device)
    b = torch.rand(1, 4, 160, 160).to(device)

    out = model(a, b)
    print(out.shape)



