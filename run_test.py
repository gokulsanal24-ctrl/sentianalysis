import csv
from sentiment_analyzer import analyze_csv_detailed
import json

res = analyze_csv_detailed('sample_data.csv', 'text')
print(json.dumps(res, indent=2))
