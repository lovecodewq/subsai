if [ ! -f /.dockerenv ]; then
    echo "This script is running outside a Docker container and will not proceed."
    exit 1
fi
Bin=scripts/burn_script_to_video.py
CONTAINER_NAME="subsai"
VIDEO_FILE="assets/video/john_danaher_advice_for_grapplers/john_danaher_advice_for_grapplers.mp4"
SUBTITLES_FILE="assets/video/john_danaher_advice_for_grapplers/john_danaher_advice_for_grapplers_en_ch.ass"

COMMAND="python3 $Bin \
-v ${VIDEO_FILE} \
-s ${SUBTITLES_FILE}"
${COMMAND}


