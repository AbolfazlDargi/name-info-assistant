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
    data = read_csv_data("colleagues (2).csv")
    closest = difflib.get_close_matches(name, list(data.keys()), n=1, cutoff=0.6)
    if not closest:
        return f"No information found for {name}."
    found_name = closest[0]
    career = data[found_name] if data[found_name] else "Career information not available"
    return f"{found_name} → {career}"

def explain_model() -> str:
    return ("I am an AI assistant that provides information about people's names and careers "
            "from a predefined dataset (CSV file).")


def extract_name_from_text(text, names_list):
    text = text.lower()
    for name in names_list:
        if all(part.lower() in text for part in name.split()):
            return name
    return None


if __name__ == "__main__":
    print("Welcome to the Name Information Assistant :)")
    print("Type something like 'Tell me about Bob Smith' or 'exit' to quit.\n")


    data = read_csv_data("colleagues.csv")
    all_names = list(data.keys())

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        name = extract_name_from_text(user_input, all_names)

        if name:
            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": name}],
                tools=[get_person_info]
            )
        else:
            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "user", "content": user_input}],
                tools=[explain_model]
            )

        print("Assistant:", response["message"]["content"])
