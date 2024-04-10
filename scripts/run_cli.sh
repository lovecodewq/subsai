python3 scripts/cli.py \
/data/videos/huberman_tools_to_accelerate_your_fitness_goals.mp4 \
--model openai/whisper \
--model-configs '{"model_type": "small.en"}' \
--translation_model_name facebook/m2m100_418M \
--source-lang English \
--target-lang Chinese