DATADIR=/home/wenqiangli/code/main/expand_data/videos
PROCESSED="${DATADIR}/huberman_tools_to_accelerate_your_fitness_goals_prcocessed"
INPUT_FILE="${PROCESSED}/huberman_tools_to_accelerate_your_fitness_goals_p_1.mp4"
OUTPUT_FILE="${PROCESSED}/huberman_tools_to_accelerate_your_fitness_goals_p_1_cn_en.mp4"
ffmpeg -i "${INPUT_FILE}" \
    -vf "subtitles=${PROCESSED}/part_0/subtitles/huberman_tools_to_accelerate_your_fitness_goals_p_1_en_ch.ass" \
    -max_muxing_queue_size 1024 -c:a copy \
    "${OUTPUT_FILE}"
