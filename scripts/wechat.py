import logging
import os
import signal
import sys
import time
import xml.etree.ElementTree as ET
import json
from pathlib import Path
import random
import openai
import itchat
from text import handler_text
from itchat.content import *

# logging.basicConfig(level=logging.INFO)
log = logging.getLogger('main')

config = {
    'default_prompt': "你是一个模仿人类聊天的机器人",
    'model': 'gpt-3.5-turbo',
    'history_len': 15,
}

config = type('Config', (object,), config)()

def stop_program(signal, frame):
    log.info('WeChatbot Closing Save some data')
    itchat.dump_login_status()
    sys.exit(0)

signal.signal(signal.SIGTERM, stop_program)

class WeChatGPT:

    def __init__(self):
        itchat.auto_login(enableCmdQR=2, hotReload=True, statusStorageDir='./cookie.bin')

        self.history = {}
        self.prompts = {}
        openai.api_key = '''sk-test'''
        openai.api_base = "http://127.0.0.1:8000/v1"
        self.kto_records_path = Path('data/kto_records.json')
        self.dpo_records_path = Path('data/dpo_records.json')
        
        log.info("init successful!")

    def handler_history(self, msg):
        self.history.setdefault(msg.user.userName, [])
        history = self.history[msg.user.userName]
        need_remove_len = len(history) - config.history_len
        if need_remove_len > 0:
            for i in range(need_remove_len):
                # 必须出一对 
                history.pop(0)
                history.pop(0)
        return history

    def save_feedback(self, user_message, assistant_message, label):
        record = {
            "messages": [
                {"content": user_message, "role": "user"},
                {"content": assistant_message, "role": "assistant"}
            ],
            "label": label
        }
        if self.kto_records_path.exists():
            with open(self.kto_records_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(record)
        
        with open(self.kto_records_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_dual_feedback(self, user_message, response_1, response_2, choice):
        record = {
            "conversations": [
                {"from": "human", "value": user_message}
            ],
            "chosen": {"from": "gpt", "value": response_1 if choice == 1 else response_2},
            "rejected": {"from": "gpt", "value": response_2 if choice == 1 else response_1}
        }
        if self.dpo_records_path.exists():
            with open(self.dpo_records_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(record)
        
        with open(self.dpo_records_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def reply(self, msg):
        if time.time() - msg.CreateTime > 5:
            return None

        history = self.handler_history(msg)
        user_message = msg.text
        res = handler_text(content=user_message, history=history, config=config)
        res = res.split('，')
        res[-1] = res[-1].replace('。', '')
        assistant_message = ''.join(res)

        if res[0] == '':
            res[0] = '机器人他无语了'
        for r in res:
            msg.user.send(r)
            time.sleep(2.2)

        # 添加到历史记录中
        history.append({"content": user_message, "role": "user"})
        history.append({"content": assistant_message, "role": "assistant"})

    def dual_reply(self, msg):
        history = self.handler_history(msg)
        user_message = msg.text
        response_1 = handler_text(content=user_message, history=history, config=config)
        response_2 = handler_text(content=user_message, history=history, config=config)

        response_1 = response_1.split('，')
        response_1[-1] = response_1[-1].replace('。', '')
        response_1 = ''.join(response_1)

        response_2 = response_2.split('，')
        response_2[-1] = response_2[-1].replace('。', '')
        response_2 = ''.join(response_2)

        msg.user.send(f"选出你更喜欢的回复：\n1: {response_1}\n2: {response_2}")
        
        # 记录双重回复，等待用户选择
        self.history[msg.user.userName].append({
            "content": user_message,
            "role": "user",
            "type": "dual_reply",
            "response_1": response_1,
            "response_2": response_2
        })

    def run(self):
        @itchat.msg_register(FRIENDS)
        def add_friend(msg):
            """自动同意好友"""
            root = ET.fromstring(msg.content)
            ticket = root.get('ticket')
            # itchat.accept_friend(msg.user.userName, ticket)

        @itchat.msg_register(TEXT)
        def friend(msg):
            """处理私聊消息"""
            log.info(f"{msg.user.NickName}: {msg.text}")

            if msg.text in ['不错', '不好']:
                last_messages = self.history.get(msg.user.userName, [])[-2:]
                if len(last_messages) == 2:
                    user_message = last_messages[0]['content']
                    assistant_message = last_messages[1]['content']
                    self.save_feedback(user_message, assistant_message, msg.text == '不错')
                msg.user.send('谢谢你的反馈！')
            elif msg.text in ['1', '2']:
                last_messages = self.history.get(msg.user.userName, [])
                if last_messages and last_messages[-1].get("type") == "dual_reply":
                    last_dual = last_messages.pop()
                    user_message = last_dual['content']
                    response_1 = last_dual['response_1']
                    response_2 = last_dual['response_2']
                    choice = int(msg.text)
                    self.save_dual_feedback(user_message, response_1, response_2, choice)
                    msg.user.send('谢谢你的选择！')
            else:
                # 50%的概率生成双重回复
                if random.random() < 0.5:
                    self.dual_reply(msg)
                else:
                    self.reply(msg)

        @itchat.msg_register(TEXT, isGroupChat=True)
        def groups(msg):
            """处理群聊消息"""
            if msg.isAt:
                self.reply(msg)

        itchat.run(debug=True)

if __name__ == "__main__":

    try:
        weChatGPT = WeChatGPT()
        weChatGPT.run()
    except KeyboardInterrupt:
        log.info("bye!")
