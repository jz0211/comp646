import torch
import torch.nn as nn
import torch.nn.functional as F

# Temporal Feature Extractor for Video Sequences
class TemporalFeatureExtractor(nn.Module):
    def __init__(self, input_nc, ngf=64, n_layers=1, bidirectional=False):
        super(TemporalFeatureExtractor, self).__init__()
        self.ngf = ngf
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3)
        # Adjusting LSTM input/output dimensions
        self.lstm = nn.LSTM(ngf * ngf, ngf, num_layers=n_layers, batch_first=True, bidirectional=bidirectional)
        self.conv2 = nn.Conv2d(ngf * (2 if bidirectional else 1), ngf, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: [batch, time, channels, height, width]
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        x = F.relu(self.conv1(x))
        x = x.view(b, t, self.ngf * h * w)  # Correct reshaping for LSTM
        x, _ = self.lstm(x)
        x = x.view(b, t, self.ngf, h, w)
        x = torch.mean(x, dim=1)  # Average over the time dimension
        x = x.view(b, self.ngf, h, w)
        x = F.relu(self.conv2(x))
        return x

class ResnetBlock(nn.Module):
    """Residual Block with two convolution layers and a skip connection."""
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            conv_block += [nn.ZeroPad2d(1)]
        else:
            raise NotImplementedError(f"padding type '{padding_type}' is not implemented")

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                       norm_layer(dim),
                       activation]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            conv_block += [nn.ZeroPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
    
# Global Generator that accepts video features and a static background
class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d):
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)
        # Modified to accept concatenated input of video features + static background
        model = [nn.ReflectionPad2d(3), nn.Conv2d(ngf + input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'reflect', norm_layer, activation)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, video_features, static_bgr):
        # Ensuring that static_bgr is expanded to the same spatial dimensions as video_features
        static_bgr = F.interpolate(static_bgr, size=video_features.shape[2:])
        combined_input = torch.cat([video_features, static_bgr], dim=1)
        return self.model(combined_input)

# Example usage and testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
temporal_feature_extractor = TemporalFeatureExtractor(input_nc=4, ngf=64).to(device)
global_generator = GlobalGenerator(input_nc=3, output_nc=3, ngf=64).to(device)

video_sequences = torch.randn(1, 10, 4, 256, 256, device=device)  # Simulated video frames
static_bgr = torch.randn(1, 3, 256, 256, device=device)  # Static background image

extracted_features = temporal_feature_extractor(video_sequences)
synthesized_background = global_generator(extracted_features, static_bgr)

print("Shape of synthesized background:", synthesized_background.shape)