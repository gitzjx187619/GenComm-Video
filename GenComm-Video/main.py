import sys
import argparse
import os


# 确保能引用 src 和 evaluation
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils import video_to_frames, save_frames_to_video
from src.encoder import SemanticEncoder
from src.decoder import GenerativeDecoder
from evaluation.simulate_channel import ChannelSimulator
from evaluation.compare_metrics import MetricsEvaluator


def main():
    # 0. 参数设置
    input_video = "test.mp4"  # 请确保根目录下有这个文件
    output_video = "output_generate.mp4"

    if not os.path.exists(input_video):
        print(f"❌ Error: {input_video} not found. Please put a video file in this directory.")
        return

    print("=== Step 1: Initialization ===")
    encoder = SemanticEncoder()
    decoder = GenerativeDecoder()
    channel = ChannelSimulator()

    print("\n=== Step 2: Encoding (Semantics + Structure) ===")
    # 读取前30帧进行演示 (为了节省跑分时间)
    frames = video_to_frames(input_video, max_frames=30)
    encoded_pkg = encoder.encode(frames)

    print("\n=== Step 3: Channel Simulation ===")
    # 计算实际占用的带宽
    transmission_stats = channel.simulate_transmission(encoded_pkg)
    # 将统计后的包（包含可能被干扰的 prompt）传给解码器
    encoded_pkg['prompt'] = transmission_stats['prompt']

    print("\n=== Step 4: Decoding (Generative Reconstruction) ===")
    reconstructed_frames = decoder.decode(encoded_pkg)
    save_frames_to_video(reconstructed_frames, output_video)

    print("\n=== Step 5: Scientific Evaluation ===")
    evaluator = MetricsEvaluator()
    # 截取原视频的前30帧存为临时文件，以便和生成的30帧对齐比较
    temp_gt = "temp_gt_short.mp4"
    save_frames_to_video(frames, temp_gt)

    evaluator.run_evaluation(
        gt_video=temp_gt,
        ours_video=output_video,
        ours_bitrate=transmission_stats['actual_bitrate_kbps']
    )

    # 清理
    if os.path.exists(temp_gt):
        os.remove(temp_gt)

    print("\n🎉 Experiment Finished! Check 'result_plot.png' and 'output_generate.mp4'.")


if __name__ == "__main__":
    main()
