import gzip
import json
import glob

# Ищем сгенерированный файл отчета
paths = glob.glob("build/**/*.opt-record.json.gz", recursive=True)
if not paths:
    print("Ошибка: файл оптимизации не найден.")
    exit(1)

path = paths[0]
print(f"Читаем файл: {path}\n")

with gzip.open(path, "rt") as f:
    data = json.load(f)

# Проверяем структуру
print(f"Корневой тип: {type(data).__name__}, длина: {len(data)}")
records = data[2]

def flatten(parts):
    result = []
    for part in parts:
        if isinstance(part, str):
            result.append(part)
        elif isinstance(part, dict):
            if "text" in part:
                result.append(str(part["text"]))
            elif "symtab_node" in part:
                result.append(str(part["symtab_node"]))
    return "".join(result)

found = False
for item in records:
    if not isinstance(item, dict): continue
    loc = item.get("location") or {}
    filename = str(loc.get("file", ""))
    
    # Проверяем, что это наш файл (без жесткой привязки к абсолютному пути)
    if "module.cpp" not in filename: continue
    
    line = loc.get("line", 0)
    msg = flatten(item.get("message", []))
    
    # Ищем только сообщения о векторизации
    if any(w in msg.lower() for w in ("vectoriz", "reduction", "loop", "unroll")):
        print(f"Строка {line}, {item.get('kind')}: {msg[:250]}")
        found = True

if not found:
    print("Не найдено сообщений о векторизации. Возможно, GCC 15 пишет их в другом формате.")
