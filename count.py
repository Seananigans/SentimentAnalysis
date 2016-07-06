import sqlite3

conn = sqlite3.connect("knowledgeBase.db")
conn.text_factory = str
c = conn.cursor()

for row in c.execute("SELECT COUNT(*) FROM imdb_reviews"):
	print row