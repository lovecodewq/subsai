from openai import OpenAI
import os
from dotenv import load_dotenv
import pysubs2

load_dotenv()  # This loads the environment variables from .env
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
ass_file = '/data/videos/huberman_tools_to_accelerate_your_fitness_goals/part_0/huberman_tools_to_accelerate_your_fitness_goals_p_0_en.ass'
out_file = '/data/videos/huberman_tools_to_accelerate_your_fitness_goals/part_0/huberman_tools_to_accelerate_your_fitness_goals_p_0_en_ch_1.ass'

placeholder='#@$%)'
def process_subtitles(input_file, output_file):
    subs = pysubs2.load(input_file)
    for item in subs:
        print(f"{item.start} --> {item.end}: {item.text}")
    print('----------------------------------------------------')
    translated_lines = translate_text(subs)
    print('----------------------------------------------------')
    for item in translated_lines:
        print(item)
    print('----------------------------------------------------')

    print("l1 ", len(subs))
    print("l2 ", len(translated_lines))

    # Create a new list to hold all subtitle events
    new_subs = pysubs2.SSAFile()
    # Iterate through the original and translated subtitles
    for sub, chinese_text in zip(subs, translated_lines):
        # Append original subtitle line (English)
        new_subs.append(sub)
        print(chinese_text)
        time_info, translated_text = chinese_text.split(': ', 1)
        # Create a new subtitle event for the Chinese translation
        translated_sub = pysubs2.SSAEvent(
            start=sub.start, end=sub.end, style=sub.style)
        translated_sub.text = translated_text
        new_subs.append(translated_sub)
    # new_subs.events.sort(key=lambda x: x.start)
    new_subs.save(output_file)


def translate_text(subtitles, source_lang='en', target_lang='zh', placeholder='@$%)'):
    PROMPT_MSG = (
        f'You will translate English to Chinese for a concatenated text of subtitles.'
        f'The user will give a concatednated text of English subtitles, which will be translated into Chinese.'
        f'Please keep strictly the time indicator, and placeholder "{placeholder}".'
        f'you should just replace the English character with translated charater in the same position in the translated text withou change anything else'
     )
    # Create the concatenated text with timestamps
    concatenated_text = placeholder.join(
        [f"{sub.start} --> {sub.end}: {sub.text}" for sub in subtitles])
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": PROMPT_MSG},
                {"role": "user", "content": concatenated_text}
            ]
        )
        translated_content = response.choices[0].message.content

        return translated_content.split(placeholder)
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return None


def translate_text2(texts):
    """Translate a list of texts from English to Chinese using OpenAI's API."""
    openai.api_key = os.getenv('OPENAI_API_KEY')
    translations = []

    # Process each text due to token limitations per request
    for text in texts:
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Translate the following English text to Chinese: {text}",
                max_tokens=5000
            )
            translations.append(response.choices[0].text.strip())
        except Exception as e:
            print(f"An error occurred during translation: {e}")
            translations.append("")

    return translations


process_subtitles(ass_file, out_file)
