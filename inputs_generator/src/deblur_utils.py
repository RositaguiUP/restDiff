# import cv2
# import torch
# from realesrgan import RealESRGANer
# from basicsr.archs.rrdbnet_arch import RRDBNet

# class RealESRGANDeblurer:
#     def __init__(self, model_name='RealESRGAN_x4plus', device='cuda'):
#         # 1. Setup the model architecture and define the weight URL
#         if model_name == 'RealESRGAN_x4plus':
#             model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
#                             num_block=23, num_grow_ch=32, scale=4)
#             netscale = 4
#             # We must provide the URL explicitly to avoid the 'NoneType' error
#             model_path = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        
#         # 2. Initialize the Upsampler
#         # This will automatically download the weights to 'weights/' folder
#         self.upsampler = RealESRGANer(
#             scale=netscale,
#             model_path=model_path,
#             model=model,
#             tile=400,        # Helps with memory on large images
#             tile_pad=10,
#             pre_pad=0,
#             half=True if device == 'cuda' else False, # Use FP16 for speed
#             device=device
#         )

#     def process(self, img_rgb):
#         # Real-ESRGAN expects BGR for the internal process
#         img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
#         # outscale=1 keeps the resolution the same but applies deblurring/denoising
#         output, _ = self.upsampler.enhance(img_bgr, outscale=1)
        
#         # Convert back to RGB for your existing pipeline
#         return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

import torch
import cv2
import numpy as np
# from basicsr.archs.nafnet_arch import NAFNet
from inputs_generator.utils.nafnet_arch import NAFNet

class MotionDeblurer:
    def __init__(self, model_path='inputs_generator/utils/NAFNet-GoPro-width64.pth', device='cuda'):
        self.device = device
        # Initialize NAFNet (GoPro dataset version is best for motion blur)
        self.model = NAFNet(img_channel=3, width=64, middle_blk_num=1, 
                            enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
        
        load_net = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=True)
        if 'params' in load_net:
            self.model.load_state_dict(load_net['params'], strict=True)
        elif 'params_ema' in load_net:
            self.model.load_state_dict(load_net['params_ema'], strict=True)
        
        self.model.to(self.device)
        self.model.eval()

    def process(self, img_rgb):
        # Normalize and convert to tensor
        img = img_rgb.astype(np.float32) / 255.
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Pad image to be multiple of 64 (required for NAFNet)
        h, w = img_tensor.shape[2:]
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            output = self.model(img_tensor)

        # Unpad and convert back
        output = output[:, :, :h, :w]
        output_img = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output_img = np.clip(output_img * 255., 0, 255).astype(np.uint8)
        
        return output_img