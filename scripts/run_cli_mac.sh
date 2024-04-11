python3 scripts/cli.py \
assets/video/sling.mp4 \
--model openai/whisper \
--model-configs '{"model_type": "small.en"}' \
--translation-model-name facebook/m2m100_418M \
--format ass \
--source-lang English \
--target-lang Chinese