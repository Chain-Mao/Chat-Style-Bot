import argparse
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

def completion_to_prompt(completion, model):
    if model == "llama3":
        return f"system\n\nuser\n{completion}\nassistant\n"
    elif model == "qwen":
        return f"system\n{completion}\n"
    elif model == "glm4":
        return f"\n{completion}\n"
    else:
        return f"system\n{completion}\n"


def messages_to_prompt(messages, model):
    prompt = ""
    if model == "llama3":
        for message in messages:
            if message.role == "system":
                prompt += f"system\n{message.content}\n"
            elif message.role == "user":
                prompt += f"user\n{message.content}\n"
            elif message.role == "assistant":
                prompt += f"assistant\n{message.content}\n"
        if not prompt.startswith("system"):
            prompt = "system\n" + prompt
        prompt = prompt + "assistant\n"

    elif model == "qwen":
        for message in messages:
            if message.role == "system":
                prompt += f"system\n{message.content}\n"
            elif message.role == "user":
                prompt += f"user\n{message.content}\n"
            elif message.role == "assistant":
                prompt += f"assistant\n{message.content}\n"
        prompt = "system\n" + prompt + "assistant\n"

    elif model == "glm4":
        for message in messages:
            if message.role == "system":
                prompt += f"\n{message.content}\n"
            elif message.role == "user":
                prompt += f"\n{message.content}\n"
            elif message.role == "assistant":
                prompt += f"\n{message.content}\n"
        prompt = "[gMASK]<sop>" + prompt + "\n"

    return prompt

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    conversations = []
    for chat in data:
        conversation = ""
        if 'instruction' in chat:
            conversation += f"朋友：{chat['instruction']}。"
        if 'output' in chat:
            conversation += f"我：{chat['output']}。"
        conversations.append(conversation)
        
    return conversations
    
def main(model):
    # Set language model and generation config
    Settings.llm = HuggingFaceLLM(
        model_name="Llama3-8B-Chinese-Chat",
        tokenizer_name="Llama3-8B-Chinese-Chat",
        context_window=30000,
        max_new_tokens=2000,
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        messages_to_prompt=lambda messages: messages_to_prompt(messages, model),
        completion_to_prompt=lambda completion: completion_to_prompt(completion, model),
        device_map="auto",
    )

    # Set embedding model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-base-zh-v1.5"
    )

    # Set the size of the text chunk for retrieval
    Settings.transformations = [SentenceSplitter(chunk_size=1024)]

    text = read_json("data/chat_records.json")
    documents = [Document(text=line) for line in text if line.strip()]
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=Settings.embed_model,
        transformations=Settings.transformations
    )

    wechat_bot_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "根据提供的聊天记录，回答问题。"
            ),
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=(
                "相关历史聊天记录如下\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "请你模仿回答者的聊天内容和风格，"
                "回答问题：{query_str}\n"
            )
        ),
    ]
    wechat_bot_template = ChatPromptTemplate(wechat_bot_msgs)

    query_engine = index.as_query_engine(text_qa_template=wechat_bot_template)

    your_query = "你的出生日期是什么时候？"
    print(query_engine.query(your_query).response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model with a specified template type.")
    parser.add_argument("--model", type=str, required=True, help="Type of the model template to use: 'llama3', 'qwen', or 'glm4'")
    args = parser.parse_args()

    main(args.model)
