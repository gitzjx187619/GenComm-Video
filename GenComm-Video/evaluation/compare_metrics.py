import os
import cv2
import torch
import lpips
import numpy as np
import subprocess
import matplotlib.pyplot as plt


class MetricsEvaluator:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"[Evaluator] Loading LPIPS metric...")
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)

    def calculate_lpips(self, path1, path2):
        """
        计算 LPIPS 感知损失
        """
        import cv2
        import torch.nn.functional as F

        # 内部帮助函数：读取视频的所有帧
        def read_frames(video_path):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"无法打开视频: {video_path}")

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            cap.release()
            return frames

        # 1. 读取两个视频的所有帧
        # 注意：这里会把所有帧读入内存。如果是长视频，建议改为逐帧读取的生成器。
        # 但对于 demo (30帧)，这样写最简单最稳。
        frames1_list = read_frames(path1)  # GT (比如 540p)
        frames2_list = read_frames(path2)  # Ours (比如 536p)

        scores = []
        min_len = min(len(frames1_list), len(frames2_list))

        # 2. 逐帧计算差异
        for i in range(min_len):
            # 预处理：转为 Tensor 并归一化到 [-1, 1]
            # lpips.im2tensor 期望的是 Numpy(H,W,C)，这正是 cv2 读出来的格式
            img1 = lpips.im2tensor(frames1_list[i]).to(self.device)
            img2 = lpips.im2tensor(frames2_list[i]).to(self.device)

            # ================= 尺寸强制对齐 =================
            # 解决 540p vs 536p 报错的问题
            if img1.shape != img2.shape:
                # 使用双线性插值将 img2 拉伸成 img1 的大小
                img2 = F.interpolate(
                    img2,
                    size=(img1.shape[2], img1.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            # ===============================================

            # 计算这一帧的分数
            # .item() 把 Tensor(0.3) 变成 Python float 0.3
            score = self.loss_fn(img1, img2)
            scores.append(score.item())

        if len(scores) == 0:
            return 1.0  # 也就是完全不相似，防止除零错误

        return sum(scores) / len(scores)

    def _to_tensor(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 127.5) - 1.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def generate_h264_anchors(self, input_video, bitrates=[50, 100, 150, 200, 300, 400]):
        anchors = []
        print("[Evaluator] Generating H.264 benchmarks...")
        for b in bitrates:
            out_name = f"anchor_{b}k.mp4"
            cmd = f"ffmpeg -i {input_video} -c:v libx264 -b:v {b}k -maxrate {b}k -bufsize {b * 2}k -y {out_name} -loglevel error"
            subprocess.run(cmd, shell=True)
            anchors.append((b, out_name))
        return anchors

    def run_evaluation(self, gt_video, ours_video, ours_bitrate):
        # 1. H.264 基线
        anchors = self.generate_h264_anchors(gt_video)
        h264_x, h264_y = [], []

        for b, path in anchors:
            score = self.calculate_lpips(gt_video, path)
            h264_x.append(b)
            h264_y.append(score)
            print(f"  H.264 @ {b}kbps -> LPIPS: {score:.4f}")
            if os.path.exists(path): os.remove(path)  # 清理

        # 2. Ours
        score_ours = self.calculate_lpips(gt_video, ours_video)
        print(f"  Ours  @ {ours_bitrate:.2f}kbps -> LPIPS: {score_ours:.4f}")

        # 3. Plot
        plt.figure(figsize=(10, 6))
        plt.plot(h264_x, h264_y, 'o--', color='gray', label='H.264 (Standard)')
        plt.scatter([ours_bitrate], [score_ours], color='red', s=200, marker='*', zorder=10, label='Ours (GenComm)')
        plt.title('Perceptual Quality (LPIPS) vs Bitrate')
        plt.xlabel('Bitrate (kbps)')
        plt.ylabel('LPIPS (Lower is Better)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("result_plot.png")
        print("✅ Chart saved to result_plot.png")
