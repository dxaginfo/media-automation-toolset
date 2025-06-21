#!/usr/bin/env python3

"""SceneValidator - Validates scene composition and continuity.

This module provides functionality to validate scenes in video and film production,
identifying continuity and composition issues using the Gemini API.
"""

import argparse
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

import google.cloud.storage as storage
import yaml
from flask import Request, jsonify
from google.api_core.exceptions import GoogleAPIError
from google.cloud import storage
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('scene_validator')

class SceneValidator:
    """Main class for scene validation operations."""

    def __init__(self, config_path: str = None):
        """Initialize the SceneValidator with configuration.

        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        self._setup_gemini_api()
        self._setup_storage()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or environment variables.

        Args:
            config_path: Path to configuration file.

        Returns:
            Dict containing configuration parameters.
        """
        config = {
            'gemini_api': {
                'api_key': os.environ.get('GEMINI_API_KEY', ''),
                'model_version': os.environ.get('GEMINI_MODEL_VERSION', 'gemini-pro-vision'),
            },
            'storage': {
                'bucket_name': os.environ.get('STORAGE_BUCKET', 'scene-validator-data'),
                'temp_folder': os.environ.get('TEMP_FOLDER', 'temp_processing'),
            },
            'validation': {
                'default_strictness': os.environ.get('DEFAULT_STRICTNESS', 'medium'),
                'timeout_seconds': int(os.environ.get('TIMEOUT_SECONDS', '30')),
                'max_scene_size_mb': int(os.environ.get('MAX_SCENE_SIZE_MB', '100')),
            }
        }

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    # Update config with file values
                    for section, values in file_config.items():
                        if section in config:
                            config[section].update(values)
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")

        return config

    def _setup_gemini_api(self):
        """Configure and initialize the Gemini API client."""
        try:
            api_key = self.config['gemini_api']['api_key']
            if not api_key:
                raise ValueError("Gemini API key is required")
                
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.config['gemini_api']['model_version'])
            logger.info(f"Initialized Gemini API with model {self.config['gemini_api']['model_version']}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise

    def _setup_storage(self):
        """Configure and initialize Google Cloud Storage client."""
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.config['storage']['bucket_name'])
            logger.info(f"Initialized Storage client with bucket {self.config['storage']['bucket_name']}")
        except Exception as e:
            logger.error(f"Failed to initialize Storage client: {e}")
            # Continue without storage if it fails
            self.storage_client = None
            self.bucket = None

    def validate(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a scene based on provided data.

        Args:
            scene_data: Dictionary containing scene information and content.

        Returns:
            Dictionary with validation results.
        """
        start_time = time.time()
        scene_id = scene_data.get('scene_id', str(uuid.uuid4()))
        validation_parameters = scene_data.get('validation_parameters', {})
        
        try:
            # Validate and preprocess input
            self._validate_input(scene_data)
            scene_content = scene_data.get('scene_content', {})
            
            # Process media if present
            media_analysis = self._analyze_media(scene_content)
            
            # Check continuity with adjacent scenes if specified
            continuity_analysis = self._check_continuity(scene_content, validation_parameters)
            
            # Analyze overall scene composition
            composition_analysis = self._analyze_composition(scene_content, validation_parameters)
            
            # Detect issues based on analyses
            issues = self._detect_issues(media_analysis, continuity_analysis, composition_analysis, validation_parameters)
            
            # Generate recommendations for fixing issues
            recommendations = self._generate_recommendations(issues, scene_content)
            
            # Prepare final result
            processing_time = time.time() - start_time
            result = {
                'scene_id': scene_id,
                'validation_result': {
                    'valid': len(issues) == 0,
                    'score': self._calculate_score(issues, scene_content),
                    'issues': issues,
                    'summary': self._generate_summary(issues, scene_content),
                    'metadata': {
                        'processing_time': processing_time,
                        'validator_version': '0.1.0',
                        'validation_timestamp': datetime.now().isoformat()
                    }
                }
            }
            return result
        except Exception as e:
            logger.error(f"Error validating scene {scene_id}: {e}")
            return {
                'scene_id': scene_id,
                'validation_result': {
                    'valid': False,
                    'score': 0.0,
                    'issues': [{
                        'issue_type': 'system',
                        'severity': 'high',
                        'description': f"Error processing scene: {str(e)}",
                        'recommendation': "Please check your input and try again."
                    }],
                    'summary': "Validation failed due to system error.",
                    'metadata': {
                        'processing_time': time.time() - start_time,
                        'validator_version': '0.1.0',
                        'validation_timestamp': datetime.now().isoformat()
                    }
                }
            }

    def _validate_input(self, scene_data: Dict[str, Any]) -> None:
        """Validate input data for required fields and formats.

        Args:
            scene_data: Dictionary containing scene information to validate.
            
        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if not scene_data.get('scene_content'):
            raise ValueError("scene_content is required")
            
        scene_content = scene_data['scene_content']
        
        # Check if at least one content type is provided
        has_content = any([
            'video_url' in scene_content,
            'image_urls' in scene_content and scene_content['image_urls'],
            'script_text' in scene_content
        ])
        
        if not has_content:
            raise ValueError("At least one content type (video_url, image_urls, or script_text) is required")

    def _analyze_media(self, scene_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze media content (video or images) using Gemini API.

        Args:
            scene_content: Dictionary containing scene media content.
            
        Returns:
            Dictionary with media analysis results.
        """
        # Implementation would download and process media files
        # For this example, we'll simulate the analysis
        if 'video_url' in scene_content:
            logger.info(f"Analyzing video: {scene_content['video_url']}")
            # Actual implementation would:  
            # 1. Download video from GCS or URL to temporary location
            # 2. Extract key frames or convert to format suitable for Gemini
            # 3. Send frames to Gemini API for analysis
            
            # Simulate analysis result
            return {
                'media_type': 'video',
                'elements_detected': ['person', 'room', 'furniture'],
                'analysis_complete': True
            }
            
        elif 'image_urls' in scene_content and scene_content['image_urls']:
            logger.info(f"Analyzing {len(scene_content['image_urls'])} images")
            # Similar steps for images
            return {
                'media_type': 'images',
                'elements_detected': ['person', 'room', 'furniture'],
                'analysis_complete': True
            }
            
        return {
            'media_type': 'none',
            'elements_detected': [],
            'analysis_complete': False
        }

    def _check_continuity(self, scene_content: Dict[str, Any], validation_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Check continuity with adjacent scenes.

        Args:
            scene_content: Dictionary containing current scene content.
            validation_parameters: Parameters controlling validation behavior.
            
        Returns:
            Dictionary with continuity analysis results.
        """
        if not validation_parameters.get('check_continuity', True):
            return {'continuity_checked': False}
            
        # Implementation would retrieve and compare with adjacent scenes
        # This is a simplified example
        metadata = scene_content.get('metadata', {})
        prev_scene_id = metadata.get('previous_scene_id')
        next_scene_id = metadata.get('next_scene_id')
        
        # Actual implementation would retrieve and analyze adjacent scenes
        return {
            'continuity_checked': True,
            'previous_scene_analyzed': prev_scene_id is not None,
            'next_scene_analyzed': next_scene_id is not None,
            'potential_issues': []
        }

    def _analyze_composition(self, scene_content: Dict[str, Any], validation_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze scene composition using Gemini API.

        Args:
            scene_content: Dictionary containing scene content.
            validation_parameters: Parameters controlling validation behavior.
            
        Returns:
            Dictionary with composition analysis results.
        """
        if not validation_parameters.get('check_composition', True):
            return {'composition_checked': False}
            
        # In a real implementation, we would send the content to Gemini API
        # for composition analysis
        if 'script_text' in scene_content:
            script_text = scene_content['script_text']
            try:
                # This would be an actual call to Gemini API in production
                # prompt = f"Analyze the following scene description for composition quality: {script_text}"
                # response = self.model.generate_content(prompt)
                # analysis = response.text
                
                # Simulated response
                analysis = "The scene appears well-composed based on the script description."
                
                return {
                    'composition_checked': True,
                    'analysis': analysis,
                    'potential_issues': []
                }
            except Exception as e:
                logger.error(f"Error analyzing composition with Gemini: {e}")
                return {
                    'composition_checked': True,
                    'analysis': "Error during composition analysis",
                    'potential_issues': [{'type': 'system', 'description': str(e)}]
                }
        
        return {
            'composition_checked': False,
            'reason': "No script text available for composition analysis"
        }

    def _detect_issues(self, 
                      media_analysis: Dict[str, Any], 
                      continuity_analysis: Dict[str, Any], 
                      composition_analysis: Dict[str, Any],
                      validation_parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect issues in the scene based on various analyses.

        Args:
            media_analysis: Results from media content analysis.
            continuity_analysis: Results from continuity checks.
            composition_analysis: Results from composition analysis.
            validation_parameters: Parameters controlling validation behavior.
            
        Returns:
            List of dictionaries containing detected issues.
        """
        issues = []
        
        # Check for issues from continuity analysis
        if continuity_analysis.get('continuity_checked', False):
            for issue in continuity_analysis.get('potential_issues', []):
                issues.append({
                    'issue_type': 'continuity',
                    'severity': issue.get('severity', 'medium'),
                    'description': issue.get('description', 'Unspecified continuity issue'),
                    'recommendation': issue.get('recommendation', 'Review scene sequence for continuity')
                })
        
        # Check for issues from composition analysis
        if composition_analysis.get('composition_checked', False):
            for issue in composition_analysis.get('potential_issues', []):
                issues.append({
                    'issue_type': 'composition',
                    'severity': issue.get('severity', 'medium'),
                    'description': issue.get('description', 'Unspecified composition issue'),
                    'recommendation': issue.get('recommendation', 'Review scene composition')
                })
        
        # Add additional checks for lighting and costume if enabled
        if validation_parameters.get('check_lighting', False):
            # Lighting check implementation would go here
            pass
            
        if validation_parameters.get('check_costume', False):
            # Costume check implementation would go here
            pass
            
        return issues

    def _generate_recommendations(self, issues: List[Dict[str, Any]], scene_content: Dict[str, Any]) -> List[str]:
        """Generate recommendations for fixing detected issues.

        Args:
            issues: List of detected issues.
            scene_content: Dictionary containing scene content.
            
        Returns:
            List of recommendation strings.
        """
        # For each issue without a recommendation, generate one
        for issue in issues:
            if 'recommendation' not in issue:
                if issue['issue_type'] == 'continuity':
                    issue['recommendation'] = "Review adjacent scenes for continuity."
                elif issue['issue_type'] == 'composition':
                    issue['recommendation'] = "Consider reframing the shot for better composition."
                elif issue['issue_type'] == 'lighting':
                    issue['recommendation'] = "Adjust lighting for consistency and proper exposure."
                elif issue['issue_type'] == 'costume':
                    issue['recommendation'] = "Check costume elements for consistency with adjacent scenes."
                else:
                    issue['recommendation'] = "Review and address the issue according to production standards."
        
        return [issue.get('recommendation') for issue in issues]

    def _calculate_score(self, issues: List[Dict[str, Any]], scene_content: Dict[str, Any]) -> float:
        """Calculate overall validation score based on issues.

        Args:
            issues: List of detected issues.
            scene_content: Dictionary containing scene content.
            
        Returns:
            Float score between 0.0 and 1.0.
        """
        if not issues:
            return 1.0
            
        # Calculate score based on number and severity of issues
        base_score = 1.0
        
        severity_weights = {
            'low': 0.1,
            'medium': 0.2,
            'high': 0.4
        }
        
        for issue in issues:
            severity = issue.get('severity', 'medium')
            base_score -= severity_weights.get(severity, 0.2)
            
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, base_score))

    def _generate_summary(self, issues: List[Dict[str, Any]], scene_content: Dict[str, Any]) -> str:
        """Generate a summary of validation results.

        Args:
            issues: List of detected issues.
            scene_content: Dictionary containing scene content.
            
        Returns:
            Summary string.
        """
        if not issues:
            return "No issues detected. Scene passes validation."
            
        num_issues = len(issues)
        severity_counts = {}
        type_counts = {}
        
        for issue in issues:
            severity = issue.get('severity', 'medium')
            issue_type = issue.get('issue_type', 'unknown')
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        
        high_severity = severity_counts.get('high', 0)
        summary_parts = [f"Detected {num_issues} issues"]
        
        if high_severity > 0:
            summary_parts.append(f"including {high_severity} high severity issues")
            
        summary_parts.append(".")
        
        # Add type breakdown
        type_summary = []
        for issue_type, count in type_counts.items():
            type_summary.append(f"{count} {issue_type}")
            
        if type_summary:
            summary_parts.append(f" Issues by type: {', '.join(type_summary)}.")
            
        return "".join(summary_parts)


def validate_scene(request: Request) -> Dict[str, Any]:
    """Cloud Function entry point for scene validation.

    Args:
        request: HTTP request containing scene data.
        
    Returns:
        JSON response with validation results.
    """
    try:
        request_json = request.get_json(silent=True)
        if not request_json:
            return jsonify({'error': 'No data provided'}), 400
            
        validator = SceneValidator()
        result = validator.validate(request_json)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in validate_scene function: {e}")
        return jsonify({'error': str(e)}), 500


def main():
    """Command-line entry point for local testing."""
    parser = argparse.ArgumentParser(description='Scene Validator')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--input', help='Path to input JSON file')
    args = parser.parse_args()
    
    try:
        validator = SceneValidator(config_path=args.config)
        
        if args.input and os.path.exists(args.input):
            with open(args.input, 'r') as f:
                scene_data = json.load(f)
        else:
            # Sample data for testing
            scene_data = {
                'scene_id': 'test-scene',
                'scene_content': {
                    'script_text': "JOHN enters the room, holding a blue coffee mug."
                }
            }
            
        result = validator.validate(scene_data)
        print(json.dumps(result, indent=2))
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        return 1
        
    return 0


if __name__ == '__main__':
    exit(main())
