import ollama
import sys
import csv
import re
import difflib
import json

def read_csv_data(file_path):
    data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for row in reader:
                if len(row) >= 2:
                    name = row[0].strip()
                    career = row[1].strip() if len(row) > 1 else ""
                    data[name] = career
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)
    return data

def get_person_info(name):
    data = read_csv_data("colleagues (2).csv")
    found_name = None
    career = "He/She’s a professional couch potato"
    names_list = list(data.keys())
    closest = difflib.get_close_matches(name, names_list, n=1, cutoff=0.6)
    if closest:
        found_name = closest[0]
        career = data[found_name] if data[found_name] else "He/She’s a professional couch potato"

    gender = "unknown"
    if name.lower().startswith(("alice", "diana", "fiona", "hannah", "julia")):
        gender = "female"
    elif name.lower().startswith(("bob", "charlie", "ethan", "george", "ian", "amohacker")):
        gender = "male"

    display_name = found_name if found_name else name

    if career != "Career information not available" and gender != "unknown":
        return f"{display_name} is a {gender} {career}."
    elif gender != "unknown":
        return f"{display_name} is a {gender}. Career information is not available."
    else:
        return f"Information about {display_name} is not available."

def explain_model():
    return ("I am an AI assistant that provides information about people's names, genders, and careers. "
            "I use a combination of predefined data and pattern recognition to determine gender. "
            "My career information comes from a predefined dataset of colleagues.")

def parse_tool_call(response_text):
    tool_pattern = r"TOOL:\s*(\w+)\s*(\{.*\})?"
    match = re.search(tool_pattern, response_text, re.DOTALL)
    if match:
        tool_name = match.group(1)
        args_text = match.group(2) or "{}"
        try:
            args = json.loads(args_text)
        except json.JSONDecodeError:
            args = {}
        return tool_name, args
    return None, None

def handle_query(query):
    client = ollama.Client(host='http://localhost:11434', proxies=None)
    system_prompt = """
You are a function calling AI. 
You MUST answer ONLY in the following format, nothing else:

TOOL: get_person_info {"name": "<PersonName>"}

Or:

TOOL: explain_model {}
"""
    try:
        response = client.chat(
            model='llama3.2',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            options={'temperature': 0.0}
        )
        text = response['message']['content']
        tool_name, args = parse_tool_call(text)
        if tool_name == "get_person_info":
            return get_person_info(args.get("name", ""))
        elif tool_name == "explain_model":
            return explain_model()
        else:
            return text
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    print("Welcome to the Name Information Assistant :)")
    print("Commands: type a name or question, 'exit' to quit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        response = handle_query(user_input)
        print("\nAssistant:", response)
