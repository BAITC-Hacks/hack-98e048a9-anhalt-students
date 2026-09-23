import re
with open('e:/hahaton/task_data/doc_tz.html', 'r', encoding='utf-8') as f:
    html = f.read()

links = re.findall(r'href=[\"\']([^\"]+)[\"\']', html)
print(f"Total links: {len(links)}")
for l in links:
    print(l)
