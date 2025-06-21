# LoopOptimizer Troubleshooting Guide

## Common Issues

### Poor Quality After Optimization

**Symptom**: The optimized loop has visible artifacts or quality degradation

**Solution**: 
- Increase the `quality_preference` parameter: `--quality 0.9` (CLI) or `quality_preference=0.9` (API)
- Use a less aggressive optimization level: `--level light` (CLI) or `optimization_level="light"` (API)
- For GIFs, try using the WebM or APNG format instead: `--format webm`

### Processing Failures with Large Files

**Symptom**: Tool crashes or times out with very large input files

**Solution**: 
- Increase system memory allocation
- Use the `--chunk-size` parameter to process in smaller segments
- Pre-scale the video to a smaller resolution before optimization:
  ```bash
  ffmpeg -i large_video.mp4 -vf "scale=1280:-1" smaller_video.mp4
  ```
- Try processing on a more powerful machine or use cloud processing

### Loop Point Detection Issues

**Symptom**: The optimized animation has a visible jump at the loop point

**Solution**:
- Manually specify loop points using `--loop-start` and `--loop-end` parameters
- Use the `--loop-type crossfade` option for smoother transitions
- Ensure the original animation has similar start and end frames
- Try creating a bounce loop with `--loop-type bounce`

### FFmpeg Not Found Error

**Symptom**: Error message stating "FFmpeg is required but not found"

**Solution**:
- Install FFmpeg:
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `apt-get install ffmpeg`
  - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- Verify FFmpeg is in your PATH by running `ffmpeg -version` in terminal

### ImageMagick Not Found (For GIF Processing)

**Symptom**: Warnings about ImageMagick fallback or poor GIF quality

**Solution**:
- Install ImageMagick for better GIF optimization:
  - macOS: `brew install imagemagick`
  - Ubuntu/Debian: `apt-get install imagemagick`
  - Windows: Download from [imagemagick.org](https://imagemagick.org/script/download.php)

## Performance Optimization

### Speeding Up Batch Processing

- Use the `--threads` parameter to utilize multiple CPU cores: `--threads 4`
- Process smaller batches if memory is limited
- Use SSD storage for temporary files by setting the temp directory in config.yaml:
  ```yaml
  processing:
    temp_directory: "/path/to/ssd/temp"
  ```

### Reducing Memory Usage

- Lower the resolution of input files before processing
- Process one file at a time instead of batch processing
- Set a lower frame rate for output files when quality allows

## Cloud Storage Issues

### Failed to Upload to Google Cloud Storage

**Symptom**: Error messages about failed uploads to Google Cloud Storage

**Solution**:
- Verify Google Cloud credentials are properly set up: `gcloud auth application-default login`
- Check that the specified bucket exists and you have write permissions
- Ensure the service account has the necessary permissions
- Verify your network connection is stable

## Still Having Problems?

If you're still experiencing issues after trying the solutions above:

1. Check the logs for detailed error messages:
   ```bash
   python -m loop_optimizer --input file.mp4 --output optimized.mp4 --debug > log.txt 2>&1
   ```

2. Create an issue on the GitHub repository with:
   - Detailed description of the problem
   - Command line or API call you're using
   - Log output or error messages
   - Information about your system (OS, Python version, etc.)
   - Sample file that demonstrates the issue (if possible)