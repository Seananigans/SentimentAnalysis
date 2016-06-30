import re
from re import sub
import time
import cookielib
from cookielib import CookieJar
import urllib2
from urllib2 import urlopen
import sqlite3

cj = CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
opener.addheaders = [("User-agent", "Mozilla/5.0")]

jaws_url = "http://www.imdb.com/title/tt0073195/reviews?start={}"
matrix_url = "http://www.imdb.com/title/tt0133093/reviews?start={}"
citizen_kane_url = "http://www.imdb.com/title/tt0033467/reviews?start={}"

conn = sqlite3.connect("knowledgeBase.db")
conn.text_factory = str
c = conn.cursor()
try:
	c.execute("CREATE TABLE imdb_reviews (review TEXT, value REAL)")
except Exception, e:
	print "Failed to create table because " + str(e)

def main():
	for i in range(0,4000, 10):
		try:
			source_code = opener.open( matrix_url.format(i)).read()
			split_source = re.findall(r'<hr size="1" noshade="1">(.*?)<hr size="1" noshade="1">', 
			source_code, re.DOTALL)[0]
			find_scores = re.findall(r'<img width="102" height="12" alt(.*?)Was the above review useful to you', 
			split_source, re.DOTALL)
			for item in find_scores:
				score = int(re.findall(r'^="(.*?)\D', item, re.DOTALL)[0])
				review = re.findall(r'<p>(.*?)</p>', item, re.DOTALL)
				review = re.sub(r'[\n]'," ", review[0])
				review = re.sub(r'<br>'," ", review).strip()
				
				query = "SELECT * FROM imdb_reviews WHERE review=?"
				c.execute(query, [(review)])
				data = c.fetchone()
				if data is None:
					print "Adding review to our database."
					c.execute("INSERT INTO imdb_reviews (review, value) VALUES (?,?)", (review, score))
					conn.commit()
				else:
					print "Review already in our database."
		except Exception, e:
			print str(e)
			print "Error in main try."
			time.sleep(1)
		time.sleep(1)
		
main()