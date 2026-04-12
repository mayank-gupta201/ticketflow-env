import os

for root, _, files in os.walk("."):
    if ".venv" in root:
        continue  #  skip virtual env

    for file in files:
        if file.endswith((".py", ".md", ".txt", ".yaml")):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            cleaned = content.encode("ascii", "ignore").decode()
            with open(path, "w", encoding="utf-8") as f:
                f.write(cleaned)

print("Cleaned project files only")