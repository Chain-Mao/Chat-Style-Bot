import os
import uvicorn
import argparse
import sys
sys.path.append('src')
from llamafactory.api.app import create_app
from llamafactory.chat import ChatModel
sys.path.append('scripts')
from config import load_config

def main():
    parser = argparse.ArgumentParser(description="Run the chat model service.")
    parser.add_argument('--model_name', type=str, default='llama3', help='Configuration name to load')
    args = parser.parse_args()

    config = load_config(args.model_name)
    chat_model = ChatModel(config)
    app = create_app(chat_model)
    print("Visit http://localhost:{}/docs for API document.".format(os.environ.get("API_PORT", 8000)))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 8000)), workers=1)

if __name__ == "__main__":
    main()
