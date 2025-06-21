# SceneValidator Troubleshooting Guide

## Common Issues

### Authentication Errors

**Symptom**: "Unauthorized" error when calling the API

**Solution**:
- Verify your Google Cloud credentials are properly set up
- Check that your Gemini API key is valid and correctly configured in `config.yaml`
- Make sure the service account has necessary permissions for Cloud Storage if using media files

### Scene Processing Timeout

**Symptom**: Request times out for large scenes

**Solution**:
- Increase the `timeout_seconds` value in your configuration
- Reduce scene size or resolution before sending for validation
- Use smaller clips for validation rather than entire scenes

### Inaccurate Continuity Detection

**Symptom**: Tool misses obvious continuity errors

**Solution**:
- Ensure proper scene sequencing (previous_scene_id and next_scene_id in metadata)
- Try increasing strictness by setting `strict_mode: true` in validation parameters
- Provide higher quality media or more detailed script text for better analysis

### Media Processing Errors

**Symptom**: "Error processing media file" or "Cannot analyze media"

**Solution**:
- Check that the media URL is accessible and the file format is supported
- Verify Cloud Storage permissions if using GCS URLs
- Convert media to a widely supported format (MP4 for video, JPEG/PNG for images)

## Logging and Debugging

To enable more detailed logging for troubleshooting:

1. Set the environment variable for verbose logging:
   ```bash
   export SCENE_VALIDATOR_LOG_LEVEL=DEBUG
   ```

2. Run the validator with debug flag:
   ```bash
   python scene_validator.py --config=config.yaml --debug
   ```

3. Check logs in Google Cloud Console when deployed as a Cloud Function.

## Still Having Issues?

If you're still experiencing problems after trying these solutions:

1. Check the [GitHub Issues](https://github.com/dxaginfo/media-automation-toolset/issues) to see if others have encountered similar problems
2. File a new issue with detailed information about the problem, including:
   - Complete error message and stack trace
   - Configuration settings (with sensitive info redacted)
   - Sample input data that causes the problem
3. For urgent assistance, contact the maintainers directly