import sqlite3, os, shutil

history_path = os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\User Data\Default\History')
temp_path = os.path.expandvars(r'%TEMP%\chrome_history_temp3.db')
shutil.copy2(history_path, temp_path)
conn = sqlite3.connect(temp_path)
c = conn.cursor()
q = "SELECT url, title FROM urls WHERE url LIKE '%spreadsheets%' OR url LIKE '%.csv%' OR url LIKE '%.xlsx%' OR url LIKE '%drive.google.com/drive%' ORDER BY last_visit_time DESC LIMIT 30"
for row in c.execute(q):
    try:
        print(f"{row[1]} -> {row[0]}")
    except:
        print(f"URL: {row[0]}")
conn.close()
