import zlib
import numpy as np
from PIL import Image


class ChannelSimulator:
    def __init__(self, target_bandwidth_kbps=50):
        self.target_bandwidth_kbps = target_bandwidth_kbps

    def estimate_bits(self, data_bytes):
        return len(data_bytes) * 8

    def compress_edge_map(self, edge_image_pil):
        """
        学术级压缩仿真：
        1. 下采样 (Downsample): 语义骨架不需要高清，缩小到 256px 足够还原结构。
        2. 位打包 (Bit Packing): 将 0/255 的 mask 转为 1bit/pixel 的二值流。
        3. 熵编码 (Entropy Coding): 使用 zlib 模拟。
        """
        # 1. 强力下采样 (例如缩小到宽 256 像素，保持比例)
        # 这是语义通信大幅节省带宽的核心原因！
        base_width = 192
        w_percent = (base_width / float(edge_image_pil.size[0]))
        h_size = int((float(edge_image_pil.size[1]) * float(w_percent)))
        img_resized = edge_image_pil.resize((base_width, h_size), Image.Resampling.NEAREST)

        # 2. 转为 Numpy 数组 (0 或 255)
        img_array = np.array(img_resized.convert('L'))

        # 3. 二值化 (Thresholding)
        # 大于 127 的变为 1，小于的变为 0
        binary_array = (img_array > 127).astype(np.uint8)

        # 4. 位打包 (Bit Packing): 8个像素 -> 1个字节
        # 这是处理二值图的标准操作，体积直接 /8
        packed_bits = np.packbits(binary_array)

        # 5. 熵编码 (Zlib)
        compressed_data = zlib.compress(packed_bits.tobytes(), level=9)

        return compressed_data

    def simulate_transmission(self, encoded_package, fps=30):
        prompt = encoded_package['prompt']
        structure_stream = encoded_package['structure_stream']
        num_frames = len(structure_stream)

        # --- 计算 Prompt 开销 ---
        # 假设 Prompt 每秒发一次 (GOP=30)，而不是每帧发
        # 这里为了简单，把 Prompt 大小摊薄到整个片段
        prompt_bits = len(prompt.encode('utf-8')) * 8

        # --- 计算结构流开销 ---
        total_structure_bits = 0
        for edge_img in structure_stream:
            compressed_data = self.compress_edge_map(edge_img)
            total_structure_bits += self.estimate_bits(compressed_data)

        # --- 汇总统计 ---
        total_bits = prompt_bits + total_structure_bits
        duration = num_frames / fps

        # 计算 kbps
        actual_bitrate_bps = total_bits / duration
        actual_bitrate_kbps = actual_bitrate_bps / 1000.0

        print(f"\n[Channel] Simulation Report:")
        print(f"  - Frames: {num_frames}")
        print(f"  - Duration: {duration:.2f}s")
        print(f"  - Total Data: {total_bits / 8 / 1024:.2f} KB")
        print(f"  - Calculated Bitrate: {actual_bitrate_kbps:.2f} kbps (Target: <50)")

        return {
            "prompt": prompt,
            "actual_bitrate_kbps": actual_bitrate_kbps,
            "structure_stream": structure_stream
        }
