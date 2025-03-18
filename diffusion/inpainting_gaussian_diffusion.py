from diffusion.respace import SpacedDiffusion
from .gaussian_diffusion import _extract_into_tensor
import torch as th

class InpaintingGaussianDiffusion(SpacedDiffusion):
    def q_sample(self, x_start, t, noise=None, model_kwargs=None):
        """
        overrides q_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape

        bs, feat, _, frames = noise.shape
        noise *= 1. - model_kwargs['y']['inpainting_mask']  # キーポーズや軌跡の部分にはノイズは加えず、それ以外の部分にノイズを加える
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
            ) # ⇒ 𝜖𝜃 (√𝛼𝑡‣𝑥0 + √1−𝛼𝑡‣𝜖, 𝑡) 
    
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        const_noise=False,
    ):
        """
        overrides p_sample to use the inpainting mask
        
        same usage as in GaussianDiffusion
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)
        noise *= 1. - model_kwargs['y']['inpainting_mask']

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise

        # if t[0] == 999 or t[0] % 100 == 0 or t[0] == 50 or t[0] == 20:
        #     import numpy as np
        #     import copy
        #     from data_loaders.humanml.scripts.motion_process import convert_humanml_to_266, recover_from_ric_from_266
        #     from data_loaders.humanml.utils.plot_script import plot_3d_motion
        #     import data_loaders.humanml.utils.paramUtil as paramUtil
        #     mean = np.load('../motion-diffusion-model/dataset/HumanML3D/Mean_266.npy')
        #     std = np.load('../motion-diffusion-model/dataset/HumanML3D/Std_266.npy')
        #     skeleton = paramUtil.t2m_kinematic_chain

        #     tmp = (sample.cpu().permute(0, 2, 3, 1) * std + mean).float()
        #     tmp = recover_from_ric_from_266(tmp, 22)
        #     tmp = tmp.view(-1, *tmp.shape[2:]).permute(0, 2, 3, 1)[0].permute(2, 0, 1).cpu().numpy()
        #     animation_save_path = "./save/noise_visualization/sample_t={}.mp4".format(t[0])
        #     plot_3d_motion(animation_save_path, skeleton, tmp, title="",
        #                 dataset='humanml', fps=20, vis_mode='gt')

        #     befnoise = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"])
        #     befnoise = (befnoise.cpu().permute(0, 2, 3, 1) * std + mean).float()
        #     befnoise = recover_from_ric_from_266(befnoise, 22)
        #     befnoise = befnoise.view(-1, *befnoise.shape[2:]).permute(0, 2, 3, 1)[0].permute(2, 0, 1).cpu().numpy()
        #     animation_save_path = "./save/noise_visualization/befnoise_t={}.mp4".format(t[0])
        #     plot_3d_motion(animation_save_path, skeleton, befnoise, title="",
        #                 dataset='humanml', fps=20, vis_mode='gt')

        #     noise_tmp = copy.deepcopy(noise)
        #     noise_tmp = (noise_tmp.cpu().permute(0, 2, 3, 1) * std + mean).float()
        #     noise_tmp = recover_from_ric_from_266(noise_tmp, 22)
        #     noise_tmp = noise_tmp.view(-1, *noise_tmp.shape[2:]).permute(0, 2, 3, 1)[0].permute(2, 0, 1).cpu().numpy()
        #     animation_save_path = "./save/noise_visualization/noise_t={}.mp4".format(t[0])
        #     plot_3d_motion(animation_save_path, skeleton, noise_tmp, title="",
        #                 dataset='humanml', fps=20, vis_mode='gt')

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}