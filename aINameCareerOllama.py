import ollama
import csv
import difflib
import json
import re

def read_csv_data(file_path):
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        next(f)
        for row in csv.reader(f):
            if len(row) >= 2:
                data[row[0].strip()] = row[1].strip()
    return data

def get_person_info(name: str) -> str:
    data = read_csv_data("colleagues.csv")
    closest = difflib.get_close_matches(name, list(data.keys()), n=1, cutoff=0.6)
    if not closest:
        return json.dumps({"error": f"No information found for {name}."})
    found_name = closest[0]
    career = data[found_name] if data[found_name] else "Career information not available"
    return json.dumps({"name": found_name, "career": career})

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_person_info",
            "description": "Get career information about a person by their name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The full name of the person (e.g. John Doe)"
                    }
                },
                "required": ["name"]
            }
        }
    }
]

client = ollama.Client(host="http://localhost:11434")

def process_with_tools(user_input):
    response = client.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": user_input}],
        tools=tools
    )
    
    message = response['message']
    
    if 'tool_calls' in message and message['tool_calls']:
        for tool_call in message['tool_calls']:
            function_name = tool_call['function']['name']
            function_args = json.loads(tool_call['function']['arguments'])
            
            if function_name == "get_person_info":

                name = function_args.get('name', '')
                function_result = get_person_info(name)
                

                follow_up_response = client.chat(
                    model="llama3.2",
                    messages=[
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                        {"role": "tool", "content": function_result, "tool_call_id": tool_call['id'], "name": function_name}
                    ]
                )
                
                return follow_up_response['message']['content']
    
    return message['content']


if __name__ == "__main__":
    print("Welcome to the Name Information Assistant :)")
    print("Type something like 'Tell me about Bob Smith' or 'exit' to quit.\n")
    
    try:
        test_response = client.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "test"}],
            tools=tools
        )
        print("✓ Function calling supported!")
    except Exception as e:
        print(f"✗ Function calling may not be supported: {e}")
        print("Please make sure you're using a model that supports function calling.")
        print("Try: llama3.2, qwen2.5, or other newer models.")
        exit()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            response = process_with_tools(user_input)
            print("Assistant:", response)
        except Exception as e:
            print(f"Error: {e}")
            data = read_csv_data("colleagues.csv")
            all_names = list(data.keys())
            name = None
            

            for potential_name in all_names:
                if all(part.lower() in user_input.lower() for part in potential_name.split()):
                    name = potential_name
                    break
            
            if name:
                result = json.loads(get_person_info(name))
                if "error" in result:
                    print("Assistant:", result["error"])
                else:
                    print("Assistant:", f"{result['name']} --> {result['career']}")
            else:
                print("Assistant: I can't Find This Name Please Try Some one Name ")