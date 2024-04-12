from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()  # This loads the environment variables from .env
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)
PROMPT_MSG = '''
you are a professional interpreter to translate a subtitle into Chinese, The user will offer the lines of the subtitles, 
and you will just need to translate the lines into Chinese, you should keep the line number and the time indicator (which has a format like HOUR:MIN:SEC,MS --> HOUR:MIN:SEC,MS)
'''
print(completion.choices[0].message)

message_content = ''.join(all_lines[line_idx:min(line_idx+CHUNK_SIZE*BATCH_SIZE, len(all_lines))])
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": PROMPT_MSG},
        {"role": "user", "content": message_content}
     ]
)
# line_idx += CHUNK_SIZE * BATCH_SIZE
result += completion.choices[0].message.content + "\n"