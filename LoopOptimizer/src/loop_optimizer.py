#!/usr/bin/env python3

"""
LoopOptimizer - Optimizes animation and video loops for efficiency.

This module provides functionality to optimize looping media files by identifying
redundant frames, optimizing transitions, and applying efficient compression.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from google.cloud import storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('loop_optimizer')


class LoopOptimizer:
    """Main class for media loop optimization operations."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the LoopOptimizer with configuration.

        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self._setup_storage()
        self._check_dependencies()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or environment variables.

        Args:
            config_path: Path to configuration file.

        Returns:
            Dict containing configuration parameters.
        """
        default_config = {
            'processing': {
                'default_quality': 0.8,
                'max_threads': 4,
                'temp_directory': '/tmp/loop-optimizer',
            },
            'storage': {
                'local_output_dir': './output',
                'use_cloud_storage': False,
                'gcs_bucket': os.environ.get('GCS_BUCKET', 'loop-optimizer'),
            },
            'api': {
                'enable_api_server': False,
                'port': 8080,
                'allowed_origins': ['http://localhost:3000'],
            },
            'optimization': {
                'default_level': 'moderate',
                'enable_ml_optimization': False,
                'ml_model_path': './models/frame_analyzer.h5',
            }
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    # Update the default config with file values
                    for section, values in file_config.items():
                        if section in default_config:
                            default_config[section].update(values)
                        else:
                            default_config[section] = values
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")

        return default_config

    def _setup_storage(self):
        """Configure and initialize storage (local and cloud)."""
        # Ensure local directories exist
        os.makedirs(self.config['processing']['temp_directory'], exist_ok=True)
        os.makedirs(self.config['storage']['local_output_dir'], exist_ok=True)

        # Setup Google Cloud Storage client if enabled
        if self.config['storage']['use_cloud_storage']:
            try:
                self.storage_client = storage.Client()
                self.bucket = self.storage_client.bucket(self.config['storage']['gcs_bucket'])
                logger.info(f"Initialized Cloud Storage with bucket {self.config['storage']['gcs_bucket']}")
            except Exception as e:
                logger.error(f"Failed to initialize Cloud Storage: {e}")
                self.config['storage']['use_cloud_storage'] = False
                logger.warning("Falling back to local storage only")

    def _check_dependencies(self):
        """Check if required external dependencies are available."""
        try:
            # Check for FFmpeg
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            logger.info("FFmpeg is available")
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.error("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH.")
            raise RuntimeError("FFmpeg is required but not found")

        # Check for OpenCV
        logger.info(f"Using OpenCV version: {cv2.__version__}")

    def optimize(self, 
                input_path: str = None, 
                output_path: str = None,
                media_source: Dict[str, Any] = None,
                optimization_parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize a media file for efficient looping.

        Args:
            input_path: Path to the input media file (local optimization).
            output_path: Path where the optimized file should be saved (local optimization).
            media_source: Dictionary containing media source information (API mode).
            optimization_parameters: Dictionary containing optimization parameters (API mode).

        Returns:
            Dictionary with optimization results.
        """
        start_time = time.time()
        media_id = str(uuid.uuid4())
        
        # Determine operation mode (local file or API)
        if input_path and os.path.exists(input_path):
            # Local file mode
            if not output_path:
                output_dir = self.config['storage']['local_output_dir']
                output_path = os.path.join(
                    output_dir, 
                    f"optimized_{os.path.basename(input_path)}"
                )
            
            # Extract media info from the local file
            media_info = self._get_media_info(input_path)
            
            # Use default optimization parameters if not provided
            if not optimization_parameters:
                optimization_parameters = {
                    "quality_preference": self.config['processing']['default_quality'],
                    "optimization_level": self.config['optimization']['default_level'],
                    "preserve_keyframes": True,
                    "output_format": "same"
                }
        elif media_source:
            # API mode with provided media source
            media_id = media_source.get('media_id', media_id)
            media_url = media_source.get('media_url')
            
            if not media_url:
                raise ValueError("media_url is required in media_source")
                
            # Download the media file to a temporary location
            input_path = self._download_media(media_url)
            
            # Extract media info
            media_info = {
                "format": media_source.get('format', self._detect_format(input_path)),
                "loop_type": media_source.get('loop_type', 'none'),
                "metadata": media_source.get('metadata', self._get_media_info(input_path))
            }
            
            # Use provided optimization parameters or defaults
            if not optimization_parameters:
                optimization_parameters = {
                    "quality_preference": self.config['processing']['default_quality'],
                    "optimization_level": self.config['optimization']['default_level'],
                    "preserve_keyframes": True,
                    "output_format": "same"
                }
                
            # Determine output path based on format
            output_format = optimization_parameters.get('output_format', 'same')
            if output_format == 'same':
                output_ext = os.path.splitext(input_path)[1]
            else:
                output_ext = f".{output_format}"
                
            output_path = os.path.join(
                self.config['processing']['temp_directory'],
                f"{media_id}_optimized{output_ext}"
            )
        else:
            raise ValueError("Either input_path or media_source must be provided")

        try:
            # Create a temporary working directory
            with tempfile.TemporaryDirectory(dir=self.config['processing']['temp_directory']) as temp_dir:
                # Process the file according to its type
                file_type = self._detect_file_type(input_path)
                
                if file_type == 'video':
                    result = self._optimize_video(input_path, output_path, optimization_parameters, temp_dir)
                elif file_type == 'animation':
                    result = self._optimize_animation(input_path, output_path, optimization_parameters, temp_dir)
                elif file_type == 'sequence':
                    result = self._optimize_sequence(input_path, output_path, optimization_parameters, temp_dir)
                else:
                    raise ValueError(f"Unsupported file type: {file_type}")
                
                # Calculate statistics
                original_size = os.path.getsize(input_path)
                optimized_size = os.path.getsize(output_path)
                size_reduction = (1 - (optimized_size / original_size)) * 100
                
                # If using cloud storage, upload the result
                optimized_url = output_path
                if self.config['storage']['use_cloud_storage']:
                    cloud_path = f"optimized/{media_id}{os.path.splitext(output_path)[1]}"
                    blob = self.bucket.blob(cloud_path)
                    blob.upload_from_filename(output_path)
                    optimized_url = blob.public_url
                
                # Prepare the result
                processing_time = time.time() - start_time
                optimization_result = {
                    "media_id": media_id,
                    "optimization_result": {
                        "success": True,
                        "optimized_url": optimized_url,
                        "statistics": {
                            "original_size": original_size,
                            "optimized_size": optimized_size,
                            "size_reduction_percent": size_reduction,
                            **result.get("statistics", {}),
                            "processing_time": processing_time
                        },
                        "optimization_details": result.get("details", {}),
                        "metadata": {
                            "timestamp": datetime.now().isoformat(),
                            "optimizer_version": "0.0.1",
                            "configuration_used": optimization_parameters
                        }
                    }
                }
                
                return optimization_result
                
        except Exception as e:
            logger.error(f"Error optimizing media {media_id}: {e}")
            return {
                "media_id": media_id,
                "optimization_result": {
                    "success": False,
                    "error": str(e),
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "optimizer_version": "0.0.1",
                        "processing_time": time.time() - start_time
                    }
                }
            }

    def _detect_file_type(self, file_path: str) -> str:
        """Detect the type of media file.

        Args:
            file_path: Path to the media file.

        Returns:
            String indicating the file type ('video', 'animation', or 'sequence').
        """
        # Check file extension first
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        # Common video formats
        if ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
            return 'video'
        
        # Common animation formats
        if ext in ['.gif', '.apng']:
            return 'animation'
        
        # For sequences, check if it's a directory with numbered images
        if os.path.isdir(file_path):
            # Check if directory contains image sequences
            image_files = [f for f in os.listdir(file_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
            if image_files:
                return 'sequence'
        
        # For other cases, try to open with OpenCV and check properties
        try:
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # If it has multiple frames, it's likely a video
                if frame_count > 1 and fps > 0:
                    return 'video'
            
            # If OpenCV can't determine it's a video, check if it's an image
            img = cv2.imread(file_path)
            if img is not None:
                # Single image - not something we can optimize as a loop
                raise ValueError("Single images cannot be optimized as loops")
        except Exception:
            pass
        
        # If we can't determine the type, default to video and let the specific optimizer handle errors
        return 'video'

    def _get_media_info(self, file_path: str) -> Dict[str, Any]:
        """Extract information about the media file.

        Args:
            file_path: Path to the media file.

        Returns:
            Dictionary containing media metadata.
        """
        media_info = {
            "frame_rate": 0.0,
            "resolution": {"width": 0, "height": 0},
            "duration": 0.0,
            "original_size": os.path.getsize(file_path)
        }
        
        try:
            # Try to get video information using OpenCV
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                media_info["frame_rate"] = cap.get(cv2.CAP_PROP_FPS)
                media_info["resolution"]["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                media_info["resolution"]["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                media_info["duration"] = cap.get(cv2.CAP_PROP_FRAME_COUNT) / media_info["frame_rate"]
                cap.release()
            else:
                # If OpenCV fails, try FFprobe
                result = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate,duration",
                        "-of", "json", file_path
                    ],
                    capture_output=True, text=True, check=True
                )
                probe_data = json.loads(result.stdout)
                
                if 'streams' in probe_data and probe_data['streams']:
                    stream = probe_data['streams'][0]
                    media_info["resolution"]["width"] = stream.get('width', 0)
                    media_info["resolution"]["height"] = stream.get('height', 0)
                    
                    # Parse frame rate (r_frame_rate is usually in the format "num/den")
                    if 'r_frame_rate' in stream:
                        try:
                            num, den = stream['r_frame_rate'].split('/')
                            media_info["frame_rate"] = float(num) / float(den)
                        except (ValueError, ZeroDivisionError):
                            pass
                    
                    media_info["duration"] = float(stream.get('duration', 0))
        except Exception as e:
            logger.warning(f"Could not extract complete media info: {e}")
        
        return media_info

    def _detect_format(self, file_path: str) -> str:
        """Detect the format of the media file.

        Args:
            file_path: Path to the media file.

        Returns:
            String indicating the media format.
        """
        _, ext = os.path.splitext(file_path)
        return ext.lower().lstrip('.')

    def _download_media(self, media_url: str) -> str:
        """Download media from a URL to a temporary file.

        Args:
            media_url: URL of the media to download.

        Returns:
            Path to the downloaded temporary file.
        """
        # For GCS URLs, use the storage client
        if media_url.startswith('gs://'):
            if not self.config['storage']['use_cloud_storage']:
                raise ValueError("Cloud Storage is not configured but a GCS URL was provided")
                
            # Parse the bucket and object path
            _, path = media_url.split('gs://', 1)
            bucket_name, object_path = path.split('/', 1)
            
            # Download the file
            bucket = self.storage_client.bucket(bucket_name)
            blob = bucket.blob(object_path)
            
            # Create a temporary file
            temp_dir = self.config['processing']['temp_directory']
            temp_file = os.path.join(temp_dir, f"temp_{uuid.uuid4()}{os.path.splitext(object_path)[1]}")
            
            blob.download_to_filename(temp_file)
            return temp_file
        else:
            # For HTTP URLs, use subprocess to call curl or wget
            temp_dir = self.config['processing']['temp_directory']
            temp_file = os.path.join(temp_dir, f"temp_{uuid.uuid4()}.bin")
            
            try:
                # Try with curl first
                subprocess.run(["curl", "-L", "-o", temp_file, media_url], check=True)
                return temp_file
            except (subprocess.SubprocessError, FileNotFoundError):
                try:
                    # Fall back to wget if curl is not available
                    subprocess.run(["wget", "-O", temp_file, media_url], check=True)
                    return temp_file
                except (subprocess.SubprocessError, FileNotFoundError):
                    raise RuntimeError("Neither curl nor wget is available for downloading")

    def _optimize_video(self, input_path: str, output_path: str, params: Dict[str, Any], temp_dir: str) -> Dict[str, Any]:
        """Optimize a video file for looping.

        Args:
            input_path: Path to the input video file.
            output_path: Path where the optimized video should be saved.
            params: Dictionary containing optimization parameters.
            temp_dir: Temporary directory for processing files.

        Returns:
            Dictionary with optimization results.
        """
        # Extract key parameters
        quality = params.get('quality_preference', self.config['processing']['default_quality'])
        level = params.get('optimization_level', self.config['optimization']['default_level'])
        target_size = params.get('target_file_size')
        preserve_keyframes = params.get('preserve_keyframes', True)
        
        # Get video information
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_duration = total_frames / fps if fps > 0 else 0
        
        # Map optimization level to frame reduction settings
        frame_reduction = {
            'light': 0.1,      # Remove up to 10% of frames
            'moderate': 0.25,  # Remove up to 25% of frames
            'aggressive': 0.4  # Remove up to 40% of frames
        }.get(level, 0.25)
        
        # Calculate target frame count
        target_frames = max(int(total_frames * (1 - frame_reduction)), 2)
        
        # Read all frames for analysis
        frames = []
        frame_importances = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Analyze frame importance
        if len(frames) > 2:
            # Calculate the difference between adjacent frames
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i], frames[i-1])
                importance = np.sum(diff)
                frame_importances.append((i, importance))
            
            # Sort frames by importance (lower importance = more similar to neighbors = can be removed)
            sorted_importances = sorted(frame_importances, key=lambda x: x[1])
            
            # Determine which frames to keep
            frames_to_remove = len(frames) - target_frames
            frames_to_remove = max(0, min(frames_to_remove, int(len(frames) * 0.9)))  # Limit max reduction
            
            # Get indices of frames to remove (least important first)
            removal_indices = [idx for idx, _ in sorted_importances[:frames_to_remove]]
            removal_indices.sort(reverse=True)  # Sort in descending order for removal
            
            # Remove the identified frames
            for idx in removal_indices:
                if idx < len(frames):
                    frames.pop(idx)
        
        # Save the optimized frames to a temporary video
        temp_video = os.path.join(temp_dir, "temp_frames.mp4")
        out = cv2.VideoWriter(
            temp_video,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Apply final compression with FFmpeg based on quality and target size
        crf_value = int(51 - (quality * 51))  # Map quality 0-1 to CRF 51-0 (lower CRF = higher quality)
        crf_value = max(18, min(28, crf_value))  # Limit CRF to reasonable range
        
        ffmpeg_cmd = [
            "ffmpeg", "-i", temp_video, "-c:v", "libx264", "-crf", str(crf_value),
            "-preset", "medium", "-an", "-y"
        ]
        
        # Add target bitrate if target size is specified
        if target_size:
            # Calculate target bitrate based on target size and duration
            if original_duration > 0:
                # Convert target size from bytes to bits, divide by duration to get bits per second
                target_bitrate = int((target_size * 8) / original_duration * 0.95)  # 95% of theoretical to allow for overhead
                ffmpeg_cmd.extend(["-b:v", f"{target_bitrate}k", "-maxrate", f"{target_bitrate * 1.5}k", "-bufsize", f"{target_bitrate * 3}k"])
        
        # Add output path
        ffmpeg_cmd.append(output_path)
        
        # Run FFmpeg for final encoding
        subprocess.run(ffmpeg_cmd, check=True)
        
        # Return optimization results
        return {
            "statistics": {
                "original_frames": total_frames,
                "optimized_frames": len(frames),
                "original_duration": original_duration,
                "optimized_duration": len(frames) / fps if fps > 0 else 0
            },
            "details": {
                "frames_removed": total_frames - len(frames),
                "compression_applied": f"libx264 CRF {crf_value}",
                "transition_improvements": "None",
                "quality_adjustments": f"Quality preference: {quality}"
            }
        }

    def _optimize_animation(self, input_path: str, output_path: str, params: Dict[str, Any], temp_dir: str) -> Dict[str, Any]:
        """Optimize an animation file (GIF, APNG) for looping.

        Args:
            input_path: Path to the input animation file.
            output_path: Path where the optimized animation should be saved.
            params: Dictionary containing optimization parameters.
            temp_dir: Temporary directory for processing files.

        Returns:
            Dictionary with optimization results.
        """
        # Extract key parameters
        quality = params.get('quality_preference', self.config['processing']['default_quality'])
        level = params.get('optimization_level', self.config['optimization']['default_level'])
        output_format = params.get('output_format', 'same')
        
        # Convert to frames first
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract frames using FFmpeg
        subprocess.run([
            "ffmpeg", "-i", input_path, "-vsync", "0",
            os.path.join(frames_dir, "frame_%04d.png")
        ], check=True)
        
        # Get list of extracted frames
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) 
                             if f.startswith("frame_") and f.endswith(".png")])
        
        if not frame_files:
            raise ValueError("Failed to extract frames from animation")
        
        # Load frames for analysis
        frames = [cv2.imread(f) for f in frame_files]
        total_frames = len(frames)
        
        # Map optimization level to frame reduction settings
        frame_reduction = {
            'light': 0.1,      # Remove up to 10% of frames
            'moderate': 0.25,  # Remove up to 25% of frames
            'aggressive': 0.4  # Remove up to 40% of frames
        }.get(level, 0.25)
        
        # Calculate frame differences for importance
        frame_importances = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i], frames[i-1])
            importance = np.sum(diff)
            frame_importances.append((i, importance))
        
        # Sort frames by importance
        sorted_importances = sorted(frame_importances, key=lambda x: x[1])
        
        # Determine frames to remove
        target_frames = max(int(total_frames * (1 - frame_reduction)), 2)
        frames_to_remove = max(0, total_frames - target_frames)
        
        # Get indices of frames to remove
        removal_indices = [idx for idx, _ in sorted_importances[:frames_to_remove]]
        removal_indices.sort(reverse=True)  # Sort in descending order for removal
        
        # Remove the identified frames from the files list
        for idx in removal_indices:
            if idx < len(frame_files):
                os.remove(frame_files[idx])
                frame_files.pop(idx)
        
        # Determine output format
        if output_format == 'same':
            output_format = os.path.splitext(input_path)[1].lstrip('.')
            if not output_format:
                output_format = 'gif'  # Default to GIF if format can't be determined
        
        # Quality setting (0-100 for GIF, varies for other formats)
        if output_format == 'gif':
            quality_arg = max(1, min(100, int(quality * 100)))
            delay = 10  # Default delay between frames (in centiseconds)
            
            # Create optimized GIF using ImageMagick
            convert_cmd = [
                "convert", "-delay", str(delay), "-loop", "0",
                "-dispose", "background", "-layers", "optimize",
                "-quality", str(quality_arg)
            ]
            convert_cmd.extend(sorted([f for f in frame_files if os.path.exists(f)]))
            convert_cmd.append(output_path)
            
            try:
                subprocess.run(convert_cmd, check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                # Fall back to FFmpeg if ImageMagick is not available
                fps = 100 / delay  # Convert centiseconds delay to fps
                subprocess.run([
                    "ffmpeg", "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, "frame_%04d.png"),
                    "-lavfi", "palettegen=stats_mode=full[pal];[0:v][pal]paletteuse=dither=sierra2_4a",
                    "-loop", "0", "-y", output_path
                ], check=True)
        else:
            # For other formats, use FFmpeg
            fps = 10  # Default fps
            
            # Format-specific settings
            if output_format == 'webm':
                # WebM with VP9
                quality_crf = max(4, min(63, int(63 - (quality * 59))))  # Map quality 0-1 to CRF 63-4
                subprocess.run([
                    "ffmpeg", "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, "frame_%04d.png"),
                    "-c:v", "libvpx-vp9", "-crf", str(quality_crf), "-b:v", "0",
                    "-loop", "0", "-an", "-y", output_path
                ], check=True)
            elif output_format == 'mp4':
                # MP4 with H.264
                quality_crf = max(18, min(28, int(28 - (quality * 10))))  # Map quality 0-1 to CRF 28-18
                subprocess.run([
                    "ffmpeg", "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, "frame_%04d.png"),
                    "-c:v", "libx264", "-crf", str(quality_crf), "-pix_fmt", "yuv420p",
                    "-loop", "0", "-an", "-y", output_path
                ], check=True)
            elif output_format == 'apng':
                # APNG format
                subprocess.run([
                    "ffmpeg", "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, "frame_%04d.png"),
                    "-f", "apng", "-plays", "0", "-y", output_path
                ], check=True)
            else:
                # Default to GIF for unknown formats
                subprocess.run([
                    "ffmpeg", "-framerate", str(fps), "-i",
                    os.path.join(frames_dir, "frame_%04d.png"),
                    "-lavfi", "palettegen=stats_mode=full[pal];[0:v][pal]paletteuse=dither=sierra2_4a",
                    "-loop", "0", "-y", output_path
                ], check=True)
        
        # Return optimization results
        return {
            "statistics": {
                "original_frames": total_frames,
                "optimized_frames": len(frame_files),
                "original_duration": total_frames / 10.0,  # Assuming 10fps as default
                "optimized_duration": len(frame_files) / 10.0
            },
            "details": {
                "frames_removed": total_frames - len(frame_files),
                "compression_applied": f"Format: {output_format}, Quality: {quality}",
                "transition_improvements": "Frame selection based on importance",
                "quality_adjustments": f"Quality preference: {quality}"
            }
        }

    def _optimize_sequence(self, input_path: str, output_path: str, params: Dict[str, Any], temp_dir: str) -> Dict[str, Any]:
        """Optimize an image sequence for looping.

        Args:
            input_path: Path to the directory containing the image sequence.
            output_path: Path where the optimized file should be saved.
            params: Dictionary containing optimization parameters.
            temp_dir: Temporary directory for processing files.

        Returns:
            Dictionary with optimization results.
        """
        # For image sequences, use similar logic as animation optimization
        # but first collect all the image files
        image_files = []
        for ext in ('.png', '.jpg', '.jpeg', '.tiff', '.bmp'):
            image_files.extend(glob.glob(os.path.join(input_path, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(input_path, f'*{ext.upper()}')))
        
        image_files.sort()  # Sort to ensure correct sequence order
        
        if not image_files:
            raise ValueError(f"No image files found in {input_path}")
        
        # Create a temporary directory for processed images
        processed_dir = os.path.join(temp_dir, "processed_sequence")
        os.makedirs(processed_dir, exist_ok=True)
        
        # Copy all images to the processing directory
        for i, img_file in enumerate(image_files):
            shutil.copy(img_file, os.path.join(processed_dir, f"frame_{i:04d}{os.path.splitext(img_file)[1]}"))
        
        # Now treat it as an animation optimization
        dummy_input = os.path.join(processed_dir, "sequence.txt")  # Just a placeholder
        with open(dummy_input, 'w') as f:
            f.write("Placeholder for sequence processing")
        
        # Modify parameters for the animation optimizer
        animation_params = params.copy()
        animation_params['input_is_sequence'] = True
        
        # Use the animation optimizer but with our processed directory
        sequence_frames = sorted(glob.glob(os.path.join(processed_dir, "frame_*.*")))
        total_frames = len(sequence_frames)
        
        # Create a temporary video from the sequence for easier processing
        temp_video = os.path.join(temp_dir, "temp_sequence.mp4")
        fps = 10  # Default fps for sequences
        
        # Get dimensions from first image
        if sequence_frames:
            img = cv2.imread(sequence_frames[0])
            if img is not None:
                height, width = img.shape[:2]
            else:
                raise ValueError("Failed to read first image in sequence")
        else:
            raise ValueError("No frames found in sequence")
        
        # Create video from sequence
        subprocess.run([
            "ffmpeg", "-framerate", str(fps), "-i",
            os.path.join(processed_dir, f"frame_%04d{os.path.splitext(sequence_frames[0])[1]}"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-y", temp_video
        ], check=True)
        
        # Now optimize the temporary video
        return self._optimize_animation(temp_video, output_path, params, temp_dir)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description='LoopOptimizer - Optimizes animation and video loops')
    parser.add_argument('--input', help='Input media file')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--batch-dir', help='Process all compatible files in directory')
    parser.add_argument('--output-dir', help='Output directory for batch processing')
    parser.add_argument('--level', choices=['light', 'moderate', 'aggressive'], 
                        default='moderate', help='Optimization aggressiveness')
    parser.add_argument('--quality', type=float, default=0.8, 
                        help='Quality preference (0.0-1.0)')
    parser.add_argument('--target-size', help='Target file size (e.g., 500KB, 2MB)')
    parser.add_argument('--format', help='Output format override')
    parser.add_argument('--loop-type', choices=['perfect', 'crossfade', 'bounce'], 
                        help='Type of loop to create')
    parser.add_argument('--threads', type=int, help='Number of processing threads')
    parser.add_argument('--use-gpu', action='store_true', help='Enable GPU acceleration if available')
    parser.add_argument('--config', help='Path to configuration file')
    args = parser.parse_args()
    
    # Initialize the optimizer
    optimizer = LoopOptimizer(config_path=args.config)
    
    # Process target size argument
    target_size = None
    if args.target_size:
        size_str = args.target_size.upper()
        multiplier = 1
        if 'KB' in size_str:
            multiplier = 1024
            size_str = size_str.replace('KB', '')
        elif 'MB' in size_str:
            multiplier = 1024 * 1024
            size_str = size_str.replace('MB', '')
        elif 'GB' in size_str:
            multiplier = 1024 * 1024 * 1024
            size_str = size_str.replace('GB', '')
        
        try:
            target_size = int(float(size_str) * multiplier)
        except ValueError:
            print(f"Error: Invalid target size format: {args.target_size}")
            return 1
    
    # Set up optimization parameters
    optimization_parameters = {
        "quality_preference": args.quality,
        "optimization_level": args.level,
        "preserve_keyframes": True,
        "output_format": args.format or "same"
    }
    
    if target_size:
        optimization_parameters["target_file_size"] = target_size
    
    if args.loop_type:
        optimization_parameters["loop_type"] = args.loop_type
    
    # Single file mode
    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return 1
        
        result = optimizer.optimize(
            input_path=args.input,
            output_path=args.output,
            optimization_parameters=optimization_parameters
        )
        
        if result["optimization_result"]["success"]:
            stats = result["optimization_result"]["statistics"]
            print(f"Optimization successful: {args.input} -> {args.output}")
            print(f"Size reduction: {stats['size_reduction_percent']:.2f}%")
            print(f"Original size: {stats['original_size'] / 1024:.2f} KB")
            print(f"Optimized size: {stats['optimized_size'] / 1024:.2f} KB")
            print(f"Frames: {stats['original_frames']} -> {stats['optimized_frames']}")
            return 0
        else:
            print(f"Optimization failed: {result['optimization_result'].get('error', 'Unknown error')}")
            return 1
    
    # Batch processing mode
    elif args.batch_dir:
        if not os.path.isdir(args.batch_dir):
            print(f"Error: Batch directory not found: {args.batch_dir}")
            return 1
        
        if not args.output_dir:
            print("Error: Output directory is required for batch processing")
            return 1
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Find all media files in the directory
        media_files = []
        for ext in ('.mp4', '.gif', '.webm', '.mov', '.avi', '.apng'):
            media_files.extend(glob.glob(os.path.join(args.batch_dir, f'*{ext}')))
            media_files.extend(glob.glob(os.path.join(args.batch_dir, f'*{ext.upper()}')))
        
        if not media_files:
            print(f"No compatible media files found in {args.batch_dir}")
            return 1
        
        print(f"Found {len(media_files)} files to process")
        
        # Process each file
        success_count = 0
        for media_file in media_files:
            base_name = os.path.basename(media_file)
            output_format = args.format or os.path.splitext(base_name)[1].lstrip('.')
            
            if args.format:
                output_name = f"{os.path.splitext(base_name)[0]}.{args.format}"
            else:
                output_name = base_name
                
            output_path = os.path.join(args.output_dir, output_name)
            
            print(f"Processing: {base_name}")
            
            try:
                result = optimizer.optimize(
                    input_path=media_file,
                    output_path=output_path,
                    optimization_parameters=optimization_parameters
                )
                
                if result["optimization_result"]["success"]:
                    success_count += 1
                    stats = result["optimization_result"]["statistics"]
                    print(f"  Reduced by {stats['size_reduction_percent']:.2f}%")
                else:
                    print(f"  Failed: {result['optimization_result'].get('error', 'Unknown error')}")
            except Exception as e:
                print(f"  Error processing {base_name}: {e}")
        
        print(f"Batch processing complete. {success_count}/{len(media_files)} files optimized successfully.")
        return 0 if success_count > 0 else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    import glob  # Required for batch processing
    exit(main())
