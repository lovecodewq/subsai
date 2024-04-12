if [ -f /.dockerenv ]; then
    echo "This script is running inside a Docker container and will not proceed."
    exit 1
fi

CONTAINER_NAME="subsai"
VIDEO_FILE="assets/video/john_danaher_advice_for_grapplers.mp4"
OUTPUT_DIR="assets/video/john_danaher_advice_for_grapplers"
COMMAND="python3 scripts/cli.py \
${VIDEO_FILE} \
--model guillaumekln/faster-whisper \
--model-configs '{\"model_type\": \"base.en\"}' \
--translation-model-name facebook/m2m100_418M \
--destination-folder ${OUTPUT_DIR} \
--format ass \
--source-lang English \
--target-lang Chinese"
# --model openai/whisper \

# Execute the command inside the Docker container
# echo "${COMMAND}"
# if ! docker exec "$CONTAINER_NAME" /bin/bash -c "$COMMAND"; then
#     echo "Failed to generate subtitles!"
#     exit 1
# fi

find "$OUTPUT_DIR" -type f -name "*.mp4" | while read -r MP4_FILE; do
    echo "MP4 file found: $MP4_FILE"  # Corrected message

    # Extracting the base name without the extension
    BASE_NAME=$(basename "$MP4_FILE" .mp4)
    ASS_FILE="$(dirname "$MP4_FILE")/${BASE_NAME}.ass"  # Added quotes

    # Check if the corresponding .ass file exists
    if [ -f "$ASS_FILE" ]; then
        echo "ASS file found: $ASS_FILE"
    else
        echo "$ASS_FILE not exist"
        continue
    fi
    OUTPUT_FILE="${OUTPUT_DIR}/${BASE_NAME}_en_ch.mp4"
    # Check if the output file already exists
    if [ -f "$OUTPUT_FILE" ]; then
        echo "$OUTPUT_FILE already exists. Skipping..."
        continue
    fi
    if ffmpeg -i "${MP4_FILE}" -vf "subtitles=${ASS_FILE}" -c:v libx264 -crf 20 -c:a copy "${OUTPUT_FILE}"; then
        echo "Successfully processed: ${OUTPUT_FILE}"
    else
        echo "Error processing $MP4_FILE"
    fi
done
