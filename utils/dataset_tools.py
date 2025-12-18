import os

def list_classes(data_dir='../data'):
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return

    print(f"Scanning {data_dir} for food classes...")
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if classes:
        print(f"Found {len(classes)} classes:")
        for c in classes:
            print(f" - {c}")
        print("\nYou can use these names to update 'utils/calorie_data.py'")
    else:
        print("No classes found. Ensure folders are named after food items.")

if __name__ == "__main__":
    list_classes()
