import ollama
import csv
import difflib

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
        return f"No information found for {name}."
    
    found_name = closest[0]
    career = data[found_name] if data[found_name] else "Career information not available"
    return f"{found_name} → {career}"

def explain_model() -> str:
    return ("I am an AI assistant that provides information about people's names and careers "
            "from a predefined dataset (CSV file).")

tools = [get_person_info, explain_model]

if __name__ == "__main__":
    print("Welcome to the Name Information Assistant :)")
    print("Commands: type a question like 'Tell me about Bob Smith', 'exit' to quit.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        response = ollama.chat(
            model="llama3.2",
            messages=[{'role': 'user', 'content': user_input}],
            tools=tools
        )

        if response['message'].get('tool_calls'):
            for tool in response['message']['tool_calls']:
                fn = tool['function']['name']
                args = tool['function']['arguments']
                if fn == "get_person_info":
                    print("\nAssistant:", get_person_info(args.get("name", "")))
                elif fn == "explain_model":
                    print("\nAssistant:", explain_model())
        else:
            print("\nAssistant:", response['message']['content'])
