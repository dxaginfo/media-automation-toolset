# SceneValidator Setup Instructions

## Prerequisites

- Google Cloud Platform account
- Gemini API access
- Python 3.9 or higher

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dxaginfo/media-automation-toolset.git
   cd media-automation-toolset/SceneValidator
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Google Cloud credentials:
   ```bash
   gcloud auth application-default login
   ```

4. Create a configuration file:
   ```bash
   cp config.example.yaml config.yaml
   ```

5. Edit the configuration file with your API keys and settings.

## Deployment

### Local Deployment

For local testing, you can run the validator locally:

```bash
cd src
python scene_validator.py --config=../config.yaml
```

### Cloud Deployment

To deploy to Google Cloud Functions:

```bash
gcloud functions deploy scene-validator \
  --runtime=python39 \
  --entry-point=validate_scene \
  --trigger-http \
  --allow-unauthenticated \
  --memory=2048MB \
  --timeout=300s
```

## Verification

After deployment, you can verify the installation by sending a test request:

```bash
curl -X POST https://us-central1-project-id.cloudfunctions.net/scene-validator \
  -H "Content-Type: application/json" \
  -d '{"scene_id": "test-scene", "scene_content": {"script_text": "JOHN enters the room."}}'
```

## Troubleshooting

See the [troubleshooting guide](./troubleshooting.md) for common issues and solutions.