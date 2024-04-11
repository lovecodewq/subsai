python3 scripts/cli.py \
/data/videos/huberman_tools_to_accelerate_your_fitness_goals.mp4 \
--model openai/whisper \
--model-configs '{"model_type": "small.en"}' \
--translation-model-name facebook/m2m100_418M \
--format ass \
--source-lang English \
--target-lang Chinese
chmod -R 777 /data/videos