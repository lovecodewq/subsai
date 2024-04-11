DATADIR=/Users/wenqiangli/code/subsai/assets/video
PROCESSED="${DATADIR}/sling_prcocessed"
INPUT_FILE="${PROCESSED}/sling_p_1.mp4"
OUTPUT_FILE="${PROCESSED}/sling_p1_en_ch.mp4"
ffmpeg -i "${INPUT_FILE}" \
    -vf "subtitles=${PROCESSED}/part_0/subtitles/sling_p_1_en_ch.ass" \
    -c:v libx264 \
    -crf 20 \
    -c:a copy \
    "${OUTPUT_FILE}"