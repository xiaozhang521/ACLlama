import json
import random

# 原始数据（可以从文件中读取，这里用字符串示例）
raw_data = open("/data/s50042884/all_data/audio-enc-moe-data/audiocaps/audiocaps-master/asr_rebuild_data.multi_task.train", "r", encoding="utf-8")

# 询问模板字典
with open("/data/s50042884/all_data/audio_caption/audiocaps/labels.json", "r") as f:
    question_data = json.load(f)
len_question_data = len(question_data)

# 读取 JSON 文件
with open("/data/s50042884/huggingface_model/libri_train_update.json", "r") as f:
    prompt_data = json.load(f)
len_prompt_data = len(prompt_data)


# 处理每一行数据
output = []
lines = raw_data.readlines()
for idx, line in enumerate(lines):
    path, description, _ = line.split("\t")

    random_question_number = random.randint(0, len_question_data - 1)
    question_line = question_data[str(random_question_number)]["prompt"].replace("<|AUDIO|>", "").strip("")

    random_prompt_number = random.randint(0, len_prompt_data - 1)
    prompt_line = prompt_data[random_prompt_number]["conversations"][1]
    user_prompt = prompt_line["value"]
    transcription = prompt_line["transcription"]
    user_prompt = user_prompt.split(transcription)[0].strip("")

    item = {
        "conversations": [
            {
                "from": "user",
                "value": question_line,
                "audio": path
            },
            {
                "from": "assistant",
                "value": f"This audio clip contains: {description}",
                # "value": f"{user_prompt} {description}",
                "transcription": description
            }
        ],
        "id": idx
    }
    
    output.append(item)

# 保存结果到文件
with open("/data/s50042884/my_code/audio_pretrain/data/audio_caps_formatted.json", "w") as f:
    json.dump(output, f, indent=2)
