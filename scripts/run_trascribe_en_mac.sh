if [ -f /.dockerenv ]; then
    echo "This script is running inside a Docker container and will not proceed."
    exit 1
fi

CONTAINER_NAME="subsai"
VIDEO_FILE="assets/video/john_danaher_advice_for_grapplers.mp4"
OUTPUT_DIR="assets/video/john_danaher_advice_for_grapplers"
COMMAND="python3 scripts/transcribe_and_translation_model_base.py \
${VIDEO_FILE} \
--model guillaumekln/faster-whisper \
--model-configs '{\"model_type\": \"base.en\"}' \
--translation-model-name facebook/m2m100_418M \
--destination-folder ${OUTPUT_DIR} \
--format srt \
--no-translate \
--source-lang English \
--target-lang Chinese \
--is-burn-to-video=False \
--is-save_merge-subtitles=False \
--split-duration 60"

# Execute the command inside the Docker container
echo "${COMMAND}"
if ! docker exec "$CONTAINER_NAME" /bin/bash -c "$COMMAND"; then
    echo "Failed to generate subtitles!"
    exit 1
fi

