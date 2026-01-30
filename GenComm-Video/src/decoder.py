import torch
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler


class GenerativeDecoder:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"[Decoder] Loading Generative Models (SD + ControlNet)...")

        # 加载 ControlNet (Canny版)
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16,
            local_files_only=True
        )

        # 加载 Stable Diffusion
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
            local_files_only=True
        ).to(self.device)

        # 开启显存优化
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()  # 关键：防止爆显存
        #self.pipe.enable_xformers_memory_efficient_attention()  # 如果安装了 xformers

    def decode(self, encoded_data):
        """
        优化版解码：关键帧复用 + 固定种子 + 强负面提示
        目标：大幅减少换装、闪烁、身份漂移
        """
        prompt = encoded_data['prompt']
        structure_stream = encoded_data['structure_stream']

        # ────────────── 1. 固定全局 Prompt ──────────────
        # 加上强约束词，告诉 AI 必须保持一致
        fixed_prompt = (
            f"{prompt}, same person throughout the entire video, "
            "consistent clothing and appearance, same outfit, no wardrobe change, "
            "identity preserved, coherent character, continuous motion"
        )

        # ────────────── 2. 超强负面提示词（防换装、防闪烁） ──────────────
        negative_prompt = """
        clothes changing, different outfit, wardrobe change, identity switch, 
        flickering, shimmering, temporal inconsistency, morphing, mutation, 
        deformed, ugly, extra limbs, bad anatomy, poorly drawn face, 
        bad proportions, blurry, low quality, overexposed, underexposed, 
        costume change, fashion switch, face morphing, inconsistent character
        """

        # ────────────── 3. 固定随机种子（让风格尽量稳定） ──────────────
        generator = torch.Generator(device=self.device).manual_seed(42)  # 42 可以换成任意固定数字

        reconstructed_frames = []
        last_output = None
        keyframe_interval = 4  # 每 8 帧生成一次新画面（可调：4=更稳但慢，12=更快但可能闪烁）

        print(f"[Decoder] 使用关键帧间隔: {keyframe_interval} 帧，固定种子: 42")

        for i, edge_map in enumerate(structure_stream):
            if i % keyframe_interval == 0:
                # 关键帧：完整生成
                print(f"  生成关键帧 {i + 1}/{len(structure_stream)} ...", end='\r')
                output = self.pipe(
                    fixed_prompt,
                    image=edge_map,
                    negative_prompt=negative_prompt,
                    num_inference_steps=25,  # 关键帧可以多步一点，质量更高
                    generator=generator,
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=0.5,  # 建议 0.6~0.8 之间，防过度约束
                ).images[0]
                last_output = np.array(output)
            else:
                # 非关键帧：直接复用上一帧（最简单有效）
                # 如果想再平滑一点，可以后续加光流 warp，但先这样已经能大幅改善
                output = last_output
                print(f"  复用关键帧 {i + 1}/{len(structure_stream)} ...", end='\r')

            reconstructed_frames.append(np.array(output))  # 确保一定是 numpy 数组


        print("\n[Decoder] 重建完成，使用关键帧复用策略。")
        return reconstructed_frames

