import numpy as np
import torch
from unimatch.unimatch import UniMatch
from torchvision.transforms.functional import hflip
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ToTensor(object):
    """Convert numpy array to torch tensor"""

    def __init__(self, no_normalize=False):
        self.no_normalize = no_normalize

    def __call__(self, sample):
        left = np.transpose(sample['left'], (2, 0, 1))  # [3, H, W]
        right = np.transpose(sample['right'], (2, 0, 1))
        sample['left'] = torch.from_numpy(left.astype(np.float32))
        sample['right'] = torch.from_numpy(right.astype(np.float32))
        return sample


class Normalize(object):
    """Normalize image, with type tensor"""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        norm_keys = ['left', 'right']

        for key in norm_keys:
            # Images have converted to tensor, with shape [C, H, W]
            for t, m, s in zip(sample[key], self.mean, self.std):
                t.sub_(m).div_(s)

        return sample
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample



class Unimatcher():
    def __init__(self, unimatch_config):

        self.model = UniMatch(
                feature_channels=unimatch_config["feature_channels"],
                num_scales=unimatch_config["num_scales"],
                upsample_factor=unimatch_config["upsample_factor"],
                num_head=unimatch_config["num_head"],
                ffn_dim_expansion=unimatch_config["ffn_dim_expansion"],
                num_transformer_layers=unimatch_config["num_transformer_layers"],
                reg_refine=unimatch_config["reg_refine"],
                task=unimatch_config["task"])

        self.padding_factor=unimatch_config["padding_factor"]
        self.inference_size=unimatch_config["inference_size"]
        self.attn_type=unimatch_config["attn_type"]
        self.attn_splits_list=unimatch_config["attn_splits_list"]
        self.corr_radius_list=unimatch_config["corr_radius_list"]
        self.prop_radius_list=unimatch_config["prop_radius_list"]
        self.num_reg_refine=unimatch_config["num_reg_refine"]
        self.pred_right_disp=unimatch_config["pred_right_disp"]
        print(unimatch_config['state_dict_path'])
        # load state_dict
        state_dict = torch.load(unimatch_config['state_dict_path'], map_location=device)
        self.model.load_state_dict(state_dict=state_dict['model'])
        self.model.to(device)

    @torch.no_grad()
    def get_disparity(self,
                         left_images,
                         right_images):
                     
        padding_factor = self.padding_factor
        inference_size = self.inference_size
        attn_type = self.attn_type
        attn_splits_list = self.attn_splits_list
        corr_radius_list = self.corr_radius_list
        prop_radius_list = self.prop_radius_list
        num_reg_refine = self.num_reg_refine
        pred_right_disp = self.pred_right_disp 
        
        self.model.eval()

        val_transform_list = [ToTensor(),
                            Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]

        val_transform = Compose(val_transform_list)
    
        fixed_inference_size = inference_size
        disparities = []

        for left, right in zip(left_images, right_images):

            sample = {'left': left*1.8, 'right': right*1.8}

            plt.imshow(np.hstack([left, right])*1.8)
            plt.show()

            sample = val_transform(sample)

            left = sample['left'].to(device).unsqueeze(0)  # [1, 3, H, W]
            right = sample['right'].to(device).unsqueeze(0)  # [1, 3, H, W]

            nearest_size = [int(np.ceil(left.size(-2) / padding_factor)) * padding_factor,
                            int(np.ceil(left.size(-1) / padding_factor)) * padding_factor]

            # resize to nearest size or specified size
            inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

            ori_size = left.shape[-2:]
            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                left = F.interpolate(left, size=inference_size,
                                    mode='bilinear',
                                    align_corners=True)
                right = F.interpolate(right, size=inference_size,
                                    mode='bilinear',
                                    align_corners=True)

            with torch.no_grad():
                if pred_right_disp:
                    left, right = hflip(right), hflip(left)

                pred_disp = self.model(left, right,
                                attn_type=attn_type,
                                attn_splits_list=attn_splits_list,
                                corr_radius_list=corr_radius_list,
                                prop_radius_list=prop_radius_list,
                                num_reg_refine=num_reg_refine,
                                task='stereo',
                                )['flow_preds'][-1]  # [1, H, W]

            if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
                # resize back
                pred_disp = F.interpolate(pred_disp.unsqueeze(1), size=ori_size,
                                        mode='bilinear',
                                        align_corners=True).squeeze(1)  # [1, H, W]
                pred_disp = pred_disp * ori_size[-1] / float(inference_size[-1])

            if pred_right_disp:
                pred_disp = hflip(pred_disp)

            disp = pred_disp[0].cpu().numpy()
            # pdb.set_trace()
            plt.imshow(disp, cmap = 'hot')
            plt.show()

            
            disparities.append(disp)         


if __name__ == "__main__":

    unimatch_config = {
        "feature_channels" : 128,
        "num_scales" : 2,
        "upsample_factor" : 4, 
        "num_head" : 1,
        "ffn_dim_expansion" : 4,
        "num_transformer_layers" : 6,
        "reg_refine" : True,
        "task" : 'stereo',
        # more config
        "padding_factor" : 32,
        "inference_size" : (768, 768),
        "attn_type" : 'self_swin2d_cross_swin1d',
        "attn_splits_list" : (2,8),
        "corr_radius_list" : (-1,4),
        "prop_radius_list" : (-1,1),
        "num_reg_refine" : 3,
        "pred_right_disp" : True,
        "state_dict_path": './gmstereo-scale2-regrefine3-resumeflowthings-mixdata.pth'
    }
    
    matcher = Unimatcher(unimatch_config)

    image = np.load('./saved_img.npy')
    left_images = []
    right_images = []
    for i in range(9):
        left_images.append(((image[0,i,...]/65536.)))
        right_images.append(((image[1,i,...]/65536.)))

    
    matcher.get_disparity(left_images, right_images)










