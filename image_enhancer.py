import os.path as osp
import glob
import cv2 as cv
import numpy as np
import torch

# if we are using the webpage
webpage = False

if webpage:
    import model.libraries.ESRGAN.RRDBNet_arch as arch
else:
    import libraries.ESRGAN.RRDBNet_arch as arch

class ESRGAN:
    def __init__(self):
        if webpage:
            # model path
            self.model_path = 'model/libraries/ESRGAN/models/RRDB_ESRGAN_x4.pth'
        else:
            # model path
            self.model_path = 'libraries/ESRGAN/models/RRDB_ESRGAN_x4.pth'

        # load the ESR-GAN model
        self.model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        self.model.load_state_dict(torch.load(self.model_path), strict=True)
        self.model.eval()
        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)

    def real_time_enhancer(self, input_image, out_path):
        # read images
        img = input_image * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(self.device)

        with torch.no_grad():
            output = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv.imwrite(out_path, output)

    def end_enhancer(self, input_img_path, out_path):
        idx = 0
        for path in glob.glob(input_img_path):
            # read images
            img = cv.imread(path, cv.IMREAD_COLOR)
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(self.device)

            with torch.no_grad():
                output = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round()
            cv.imwrite(out_path + str(idx) + '.jpg', output)

            # increment the index value
            idx += 1



