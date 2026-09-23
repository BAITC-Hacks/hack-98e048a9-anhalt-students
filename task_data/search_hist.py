import sqlite3, os, shutil

history_path = os.path.expandvars(r'%LOCALAPPDATA%\Google\Chrome\User Data\Default\History')
temp_path = os.path.expandvars(r'%TEMP%\chrome_history_temp2.db')
shutil.copy2(history_path, temp_path)
conn = sqlite3.connect(temp_path)
c = conn.cursor()
q = "SELECT url, title FROM urls WHERE url LIKE '%drive.google%' OR url LIKE '%docs.google%' OR url LIKE '%t.me%' OR url LIKE '%download%' OR url LIKE '%samruk%' OR url LIKE '%kazyna%' ORDER BY last_visit_time DESC LIMIT 40"
for row in c.execute(q):
    print(f"{row[1]} -> {row[0]}")
conn.close()
