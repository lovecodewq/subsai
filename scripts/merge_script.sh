DATADIR=/data/videos/huberman_tools_to_accelerate_your_fitness_goals
PROCESSED="${DATADIR}/part_0"
INPUT_FILE="${DATADIR}/part_0/huberman_tools_to_accelerate_your_fitness_goals_p_0.mp4"
OUTPUT_FILE="${DATADIR}/huberman_tools_to_accelerate_your_fitness_goals_p_0_cn_en.mp4"
/usr/local/bin/ffmpeg -i "${INPUT_FILE}" \
    -vf "subtitles=${DATADIR}/part_0/huberman_tools_to_accelerate_your_fitness_goals_p_0_en_ch.ass" \
    -c:v libx264 \
    -crf 20 \
    -max_muxing_queue_size 1024 -c:a copy \
    "${OUTPUT_FILE}"
#     if ffmpeg -i "${MP4_FILE}" -vf "subtitles=${ASS_FILE}" -c:v libx264 -crf 20 -c:a copy "${OUTPUT_FILE}"; then