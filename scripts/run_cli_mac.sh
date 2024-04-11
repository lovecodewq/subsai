if [ -f /.dockerenv ]; then
    echo "This script is running inside a Docker container and will not proceed."
    exit 1
fi

CONTAINER_NAME="subsai"
COMMAND="python3 scripts/cli.py \
assets/video/sling.mp4 \
--model openai/whisper \
--model-configs '{\"model_type\": \"small.en\"}' \
--translation-model-name facebook/m2m100_418M \
--format ass \
--source-lang English \
--target-lang Chinese"

# Execute the command inside the Docker container
echo "${COMMAND}"
if ! docker exec "$CONTAINER_NAME" /bin/bash -c "$COMMAND"; then
    echo "Failed to generate subtitles!"
    exit 1
fi

DATADIR=/Users/wenqiangli/code/subsai/assets/video
PROCESSED="${DATADIR}/sling_processed"
INPUT_FILE="${PROCESSED}/part_0/sling_p_0.mp4"
OUTPUT_FILE="${PROCESSED}/part_0/sling_p_0_en_ch.mp4"
ffmpeg -i "${INPUT_FILE}" \
    -vf "subtitles=${PROCESSED}/part_0/sling_p_0_en_ch.ass" \
    -c:v libx264 \
    -crf 20 \
    -c:a copy \
    "${OUTPUT_FILE}"