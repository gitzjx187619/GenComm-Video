
# GenComm-Video: LLM-Guided Semantic Video Communication 
##  Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/gitzjx187619/GenComm-Video.git
   cd GenComm-Video
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   *Recommendation: A GPU with at least 6GB VRAM (NVIDIA RTX 3060 or higher).*

3. **download modules**
   
   ```bash
   models--lllyasviel--sd-controlnet-canny
   models--runwayml--stable-diffusion-v1-5
   models--Salesforce--blip-image-captioning-base
   ```
---

##  Quick Start

1. **Prepare Input Video**:
   Place your test video (e.g., `test.mp4`) in the root directory.
   *Recommended resolution: 540p or 720p. The code will auto-resize it for AI stability.*

2. **Run the Pipeline**:

   ```bash
   python main.py
   ```

3. **Check Outputs**:

   * `output_generative.mp4`: The AI-reconstructed high-fidelity video.
   * `result_plot.png`: The R-D comparison curve against H.264.

---

##  System Architecture

```mermaid
graph LR
    A[Input Video] --> B(Encoder)
    B -->|Semantics| C[BLIP -> Text Prompt]
    B -->|Structure| D[Canny -> Edge Map]
    C --> E((Channel < 50kbps))
    D --> E
    E --> F(Decoder)
    F -->|ControlNet| G[Stable Diffusion]
    G --> H[Output Video]
```
---
##  Project Structure

```text
GenComm-Video/
├── src/
│   ├── encoder.py          # VLM Semantic Extraction & Edge Detection
│   ├── decoder.py          # Generative Reconstruction (SD + ControlNet)
│   └── utils.py            # Video I/O helpers
├── evaluation/
│   ├── simulate_channel.py # Bitrate calculation & Bit-packing simulation
│   └── compare_metrics.py  # LPIPS evaluation & Plotting
├── main.py                 # Main entry point
└── requirements.txt        # Python dependencies
```

