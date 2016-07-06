import urllib2
from urllib2 import urlopen
import re
import cookielib
from cookielib import CookieJar
import datetime
import time
import sqlite3

cj = CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
opener.addheaders = [("User-agent","Mozilla/5.0")]

conn = sqlite3.connect("knowledgeBase.db")
conn.text_factory = str
c = conn.cursor()

def create_word_valueDB(table_name):
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [str(a[0]) for a in c.fetchall()]
    print table_names
    if not (table_name in set(table_names)):
        c.execute("CREATE TABLE {} (word TEXT, value REAL)".format(table_name))
        table_names.append(table_name)
    else:
        print "Table already in database."
        
create_word_valueDB("wordVals")

def first():
	query = "SELECT * FROM synonyms_complete WHERE word=?"
	c.execute(query, [(starting_word)])
	data = c.fetchone()
	if data is None:
		print "Let's add \"{}\" to our database.".format(starting_word)
		c.execute("INSERT INTO synonyms_complete (word) VALUES (?)", (starting_word,))
		conn.commit()
		return False
	else:
		print "Word \"{}\" already here.".format(starting_word)
		return True


def main():
    try:
        page = "http://www.thesaurus.com/browse/"+starting_word+"?s=t"
        source_code = opener.open(page).read()
        try:
            synonym_source_split = source_code.split("<div class=\"synonyms-horizontal-divider\"></div>")
            x = 0
            while x<len(synonym_source_split)-1:
                try:
                    time.sleep(1)
                    synonym_split = synonym_source_split[x].split("</span></td>")[0]
                    synonym_split = synonym_split.split("class=\"synonym-description\">")
                    if len(synonym_split)>0:
                        synonym_split = synonym_split[1]
                    else:
                        synonym_split = synonym_split[0]
                    synoNyms = re.findall(r"class\=\"text\">(.*?)</span>", synonym_split)
                    for syn in synoNyms:
                    	if syn==starting_word: continue
                        query = "SELECT * FROM wordVals WHERE word=?"
                        c.execute(query, [(syn)])
                        data = c.fetchone()
                        if data is None:
                            print "Let's add \"{}\" to our database.".format(syn)
                            c.execute("INSERT INTO wordVals (word, value) VALUES (?,?)", (syn, starting_word_val))
                            conn.commit()
                        else: 
                            print "Word \"{}\" already here.".format(syn)
                except Exception, e:
                    print "Failed in 3rd try\n"+str(e)
                x += 1
    
        except Exception, e:
            print "Failed in 2nd try\n"+str(e)
            
    except Exception, e:
        print "Failed in 1st try\n"+str(e)

# main()

query = "SELECT * FROM wordVals"
words = []
vals = []
for row in c.execute(query):
	words.append(row[0])
	vals.append(row[1])
	
for word,val in zip(words,vals):
	starting_word = word
	starting_word_val = val
	if first(): continue
	main()
	
for row in c.execute("SELECT * FROM synonyms_complete"): print row