import pandas as pd
import json
import argparse
from datetime import datetime, timedelta

# 解析命令行参数
parser = argparse.ArgumentParser(description='Process chat messages and convert to JSON format.')
parser.add_argument('--input_csv', type=str, required=True, help='Input CSV file path')
parser.add_argument('--output_json', type=str, required=True, help='Output JSON file path')
args = parser.parse_args()

# 读取CSV文件
df = pd.read_csv(args.input_csv)

# 过滤Type不为1的行
df = df[df['Type'] == 1]
# 删除Remark为空的行数据
df = df[df['Remark'].notna()]
# 转换时间戳为datetime对象
df['CreateTime'] = pd.to_datetime(df['CreateTime'], unit='s')
# 排序数据
df = df.sort_values(by=['CreateTime'])

# 定义一个合并消息的函数
def merge_messages(group):
    merged = []
    last_sender = None
    last_time = None
    content = ""

    for index, row in group.iterrows():
        if last_sender is None or row['Sender'] != last_sender or (row['CreateTime'] - last_time) > timedelta(hours=1):
            if last_sender is not None:
                merged.append({'Sender': last_sender, 'Content': content.strip(), 'IsSender': last_is_sender})
            last_sender = row['Sender']
            last_time = row['CreateTime']
            content = row['StrContent'] + " "
            last_is_sender = row['IsSender']
        else:
            content += row['StrContent'] + " "
            last_time = row['CreateTime']

    if last_sender is not None:
        merged.append({'Sender': last_sender, 'Content': content.strip(), 'IsSender': last_is_sender})

    return merged

# 按日期分组
df['Date'] = df['CreateTime'].dt.date
groups = df.groupby('Date')

# 生成最终的JSON格式数据
result = []
for date, group in groups:
    merged_messages = merge_messages(group)

    i = 0
    while i < len(merged_messages) - 1:
        if merged_messages[i]['IsSender'] == 0 and merged_messages[i + 1]['IsSender'] == 1:
            user_message = merged_messages[i]
            assistant_message = merged_messages[i + 1]
            chat_record = {
                "instruction": user_message['Content'],
                "input": "",
                "output": assistant_message['Content'],
                "history": []
            }
            result.append(chat_record)
            i += 2
        else:
            i += 1

# 将结果保存为JSON文件
with open(args.output_json, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
