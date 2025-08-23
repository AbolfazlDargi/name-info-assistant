import ollama
import csv
import difflib
import json

CSV_FILE = "colleagues.csv"

def read_csv_data(file_path):
    data = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            next(f, None)  # skip header
            for row in csv.reader(f):
                if len(row) >= 2:
                    data[row[0].strip()] = row[1].strip()
    except FileNotFoundError:
        print(f"Error: CSV file '{file_path}' not found.")
    return data

def get_person_info(name: str) -> str:
    data = read_csv_data(CSV_FILE)
    closest = difflib.get_close_matches(name, list(data.keys()), n=1, cutoff=0.6)
    if not closest:
        return f"No information found for {name}."
    found_name = closest[0]
    career = data.get(found_name, "Career information not available")
    return f"{found_name} --> {career}"

def list_all_names() -> str:
    data = read_csv_data(CSV_FILE)
    return "\n".join(data.keys())

def get_person_gender(name: str) -> str:
    first_name = name.strip().split()[0]
    
    # نمونه لیست اسامی اولیه
    female_names = {"Alice", "Mary", "Linda", "Sophia", "Emma"}
    male_names = {"Bob", "John", "Michael", "David", "James"}

    if first_name in female_names:
        gender = "Female"
    elif first_name in male_names:
        gender = "Male"
    else:
        # اگر اسم در لیست نبود، از مدل کمک می‌گیریم
        client = ollama.Client(host="http://localhost:11434")
        prompt = f"Guess the gender (Male/Female) of a person with the first name '{first_name}'. Just answer Male or Female."
        response = client.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )
        gender = response['message']['content'].strip()
    
    return f"{name} --> {gender}"

def explain_model() -> str:
    return ("I am an AI assistant that provides information about people's names, careers, "
            "and can also guess gender based on the first name using AI or predefined rules.")

# تعریف ابزارها برای Ollama با متادیتای کامل
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_person_info",
            "description": "Get career information about a person by their name",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Full name of the person"}},
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_all_names",
            "description": "List all employee names in the database",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_person_gender",
            "description": "Get gender information about a person by their name",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Full name of the person"}},
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "explain_model",
            "description": "Explain what this AI assistant can do",
            "parameters": {"type": "object", "properties": {}}
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
    tool_calls = message.get('tool_calls', [])
    if not tool_calls:
        return message.get('content', "No response.")

    outputs = []
    for tool_call in tool_calls:
        function_name = tool_call['function']['name']
        args = tool_call['function'].get('arguments', {})
        if isinstance(args, str):
            args = json.loads(args)

        if function_name == "get_person_info":
            outputs.append(get_person_info(args.get("name", "")))
        elif function_name == "list_all_names":
            outputs.append(list_all_names())
        elif function_name == "get_person_gender":
            outputs.append(get_person_gender(args.get("name", "")))
        elif function_name == "explain_model":
            outputs.append(explain_model())
        else:
            outputs.append(f"Unknown tool: {function_name}")

    return "\n".join(outputs)

if __name__ == "__main__":
    print("Welcome to the Name Information Assistant :)")
    print("Type commands like 'Tell me about Bob Smith', 'List all employees', "
          "'Tell me the gender of Alice', or 'exit' to quit.\n")

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
            data = read_csv_data(CSV_FILE)
            all_names = list(data.keys())
            closest = difflib.get_close_matches(user_input, all_names, n=1, cutoff=0.6)
            if closest:
                print("Assistant:", get_person_info(closest[0]))
            elif "all" in user_input.lower() and "name" in user_input.lower():
                print("Assistant:\n" + list_all_names())
            else:
                print("Assistant: I can't find this name. Please try again.")
