import json
import argparse

def replace_variables_in_json(file_path, name, author):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        item['output'] = item['output'].replace('{{name}}', name).replace('{{author}}', author)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replace variables in JSON file in place.')
    parser.add_argument('--name', required=True, help='Name to replace in JSON.')
    parser.add_argument('--author', required=True, help='Author to replace in JSON.')
    parser.add_argument('--file_path', required=True, help='JSON file path.')

    args = parser.parse_args()
    replace_variables_in_json(args.file_path, args.name, args.author)
