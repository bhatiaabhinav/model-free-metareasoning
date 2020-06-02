def get_csv_dictionary(file_path):
    with open(file_path) as file:
        entries = [line.strip().split(",") for line in file.readlines()]
        return {entry[0].lower(): float(entry[1]) for entry in entries}
