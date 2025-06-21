# LoopOptimizer Setup Instructions

## Prerequisites

- Python 3.8 or higher
- FFmpeg 4.0+ installed and in PATH
- Google Cloud account (for cloud deployment)
- 8GB+ RAM for processing high-resolution videos

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dxaginfo/media-automation-toolset.git
   cd media-automation-toolset/LoopOptimizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install FFmpeg (if not already installed):
   - **macOS**: `brew install ffmpeg`
   - **Ubuntu/Debian**: `apt-get install ffmpeg`
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

4. Set up Google Cloud credentials (for cloud deployment):
   ```bash
   gcloud auth application-default login
   ```

## Configuration

Create a `config.yaml` file with the following settings:

```yaml
processing:
  default_quality: 0.8
  max_threads: 4
  temp_directory: "/tmp/loop-optimizer"
  
storage:
  local_output_dir: "./output"
  use_cloud_storage: false
  gcs_bucket: "my-loop-optimizer-bucket"
  
api:
  enable_api_server: false
  port: 8080
  allowed_origins: ["http://localhost:3000"]
  
optimization:
  default_level: "moderate"
  enable_ml_optimization: true
  ml_model_path: "./models/frame_analyzer.h5"
```

## Running the Tool

### Command Line Interface

```bash
# Basic optimization with default settings
python loop_optimizer.py --input video.mp4 --output optimized.mp4

# Aggressive optimization targeting a specific file size
python loop_optimizer.py --input animation.gif --output small.gif --level aggressive --target-size 500KB

# Batch processing a directory of files
python loop_optimizer.py --batch-dir ./animations --output-dir ./optimized --format webm
```

### As a Python Library

```python
from loop_optimizer import LoopOptimizer

# Initialize optimizer
optimizer = LoopOptimizer(config_path="config.yaml")

# Single file optimization
result = optimizer.optimize(
    input_path="animation.gif",
    output_path="optimized.gif",
    quality_preference=0.9,
    optimization_level="moderate"
)

print(f"Reduced size by {result['statistics']['size_reduction_percent']}%")
```

## Troubleshooting

See the [troubleshooting guide](./troubleshooting.md) for common issues and solutions.