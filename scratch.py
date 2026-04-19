import csv

with open('analyzer.csv', 'r', encoding='utf-8-sig') as f:
    content = f.read()
    print("Raw content:")
    print(repr(content[:100]))

with open('analyzer.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    print("\nFieldnames:")
    print(reader.fieldnames)
    print("\nRows:")
    for row in reader:
        print(row)
