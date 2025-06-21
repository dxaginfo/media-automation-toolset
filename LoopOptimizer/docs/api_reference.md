# LoopOptimizer API Reference

## Python API

### Basic Usage

```python
from loop_optimizer import LoopOptimizer

# Initialize the optimizer
optimizer = LoopOptimizer(config_path="config.yaml")

# Optimize a media file
result = optimizer.optimize(
    input_path="animation.gif",
    output_path="optimized.gif",
    optimization_parameters={
        "quality_preference": 0.8,
        "optimization_level": "moderate",
        "preserve_keyframes": True,
        "output_format": "webm"
    }
)

# Check results
if result["optimization_result"]["success"]:
    print(f"Optimization successful!")
    print(f"Size reduction: {result['optimization_result']['statistics']['size_reduction_percent']:.2f}%")
    print(f"Output file: {result['optimization_result']['optimized_url']}")
else:
    print(f"Optimization failed: {result['optimization_result'].get('error', 'Unknown error')}")
```

### Class Reference

#### `LoopOptimizer`

**Constructor**

```python
LoopOptimizer(config_path: Optional[str] = None)
```

- `config_path`: Path to YAML configuration file (optional)

**Methods**

```python
optimize(
    input_path: str = None, 
    output_path: str = None,
    media_source: Dict[str, Any] = None,
    optimization_parameters: Dict[str, Any] = None
) -> Dict[str, Any]
```

Parameters:
- `input_path`: Path to the input media file (local optimization)
- `output_path`: Path where optimized file should be saved (optional, auto-generated if not provided)
- `media_source`: Dictionary containing media source information (for API mode)
- `optimization_parameters`: Dictionary of optimization settings

Returns:
- Dictionary with optimization results (see Output Schema below)

## REST API

When run with the API server enabled, LoopOptimizer provides the following endpoints:

### `POST /api/v1/optimize`

Optimizes a media file based on provided parameters.

**Request Body**

```json
{
  "media_id": "string (optional)",
  "media_source": {
    "media_url": "string",
    "media_type": "string (video|animation|sequence)",
    "format": "string (mp4|gif|webm|png_sequence|etc)",
    "loop_type": "string (perfect|crossfade|bounce|none)",
    "metadata": {
      "frame_rate": "float (optional)",
      "resolution": {
        "width": "integer (optional)",
        "height": "integer (optional)"
      },
      "duration": "float (seconds, optional)",
      "original_size": "integer (bytes, optional)"
    }
  },
  "optimization_parameters": {
    "target_file_size": "integer (bytes, optional)",
    "target_duration": "float (seconds, optional)",
    "quality_preference": "float (0-1, default: 0.8)",
    "preserve_keyframes": "boolean (default: true)",
    "optimization_level": "string (light|moderate|aggressive, default: moderate)",
    "output_format": "string (same|mp4|webm|gif|apng, default: same)"
  }
}
```

**Response**

```json
{
  "media_id": "string",
  "optimization_result": {
    "success": "boolean",
    "optimized_url": "string",
    "statistics": {
      "original_size": "integer (bytes)",
      "optimized_size": "integer (bytes)",
      "size_reduction_percent": "float",
      "original_duration": "float (seconds)",
      "optimized_duration": "float (seconds)",
      "original_frames": "integer",
      "optimized_frames": "integer",
      "processing_time": "float (seconds)"
    },
    "optimization_details": {
      "frames_removed": "integer",
      "compression_applied": "string",
      "transition_improvements": "string",
      "quality_adjustments": "string"
    },
    "metadata": {
      "timestamp": "string (ISO format)",
      "optimizer_version": "string",
      "configuration_used": "object (copy of input parameters)"
    }
  }
}
```

### `GET /api/v1/jobs/{job_id}`

Returns the status and results of an asynchronous optimization job.

**Path Parameters**
- `job_id`: ID of the optimization job

**Response**

```json
{
  "job_id": "string",
  "status": "string (pending|processing|completed|failed)",
  "progress": "float (0-1)",
  "result": "object (optimization_result if completed)",
  "error": "string (if failed)"
}
```

## Command Line Interface

```
Usage: loop_optimizer.py [OPTIONS]

Options:
  --input FILE               Input media file
  --output FILE              Output file path
  --batch-dir DIRECTORY      Process all compatible files in directory
  --output-dir DIRECTORY     Output directory for batch processing
  --level [light|moderate|aggressive]
                             Optimization aggressiveness
  --quality FLOAT            Quality preference (0.0-1.0)
  --target-size SIZE         Target file size (e.g., 500KB, 2MB)
  --format FORMAT            Output format override
  --loop-type [perfect|crossfade|bounce]
                             Type of loop to create
  --threads INTEGER          Number of processing threads
  --use-gpu                  Enable GPU acceleration if available
  --config PATH              Path to configuration file
  --help                     Show this message and exit
```

## Configuration File Reference

The configuration file uses YAML format with the following structure:

```yaml
processing:
  default_quality: 0.8         # Default quality preference (0.0-1.0)
  max_threads: 4               # Maximum processing threads
  temp_directory: "/tmp/loop-optimizer"  # Directory for temporary files
  
storage:
  local_output_dir: "./output"  # Directory for output files
  use_cloud_storage: false      # Whether to use Google Cloud Storage
  gcs_bucket: "bucket-name"     # GCS bucket name
  
api:
  enable_api_server: false      # Whether to run the REST API server
  port: 8080                    # API server port
  allowed_origins: ["http://localhost:3000"]  # CORS settings
  
optimization:
  default_level: "moderate"     # Default optimization level
  enable_ml_optimization: false # Whether to use ML for optimization
  ml_model_path: "./models/frame_analyzer.h5"  # ML model path
```