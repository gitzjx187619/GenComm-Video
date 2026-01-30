import cv2
import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


class SemanticEncoder:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"[Encoder] Loading VLM (BLIP) on {self.device}...")

        # 加载轻量级图生文模型
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",local_files_only=True)
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",local_files_only=True).to(
            self.device)

    def extract_semantics(self, frame_rgb):
        """利用大模型提取语义 Prompt"""
        raw_image = Image.fromarray(frame_rgb)
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=50)

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        # 增强 Prompt，保证生成质量
        enhanced_prompt = f"{caption}, masterpiece, best quality, 4k, realistic, cinematic lighting"
        return enhanced_prompt

    def extract_structure(self, frame_rgb):
        """提取 Canny 边缘作为结构骨架"""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        # 阈值可调，决定了保留多少细节
        edges = cv2.Canny(gray, 50, 150)
        # 扩展为3通道 (ControlNet 需要)
        edges_3c = np.stack([edges, edges, edges], axis=2)
        return Image.fromarray(edges_3c)

    def encode(self, video_frames):
        """
        编码流程：
        1. 语义流：提取关键帧的文本描述
        2. 结构流：提取每一帧的边缘图
        """
        print(f"[Encoder] Processing {len(video_frames)} frames...")

        # 简化策略：取中间帧生成全局语义描述
        mid_idx = len(video_frames) // 2
        global_prompt = self.extract_semantics(video_frames[mid_idx])
        print(f"[Encoder] Generated Semantic Prompt: '{global_prompt}'")

        structural_stream = []
        for frame in video_frames:
            edge_map = self.extract_structure(frame)
            structural_stream.append(edge_map)

        return {
            "prompt": global_prompt,
            "structure_stream": structural_stream,
            "original_shape": video_frames[0].shape
        }
