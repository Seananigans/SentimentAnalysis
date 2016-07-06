import re
from re import sub
import time
import cookielib
from cookielib import CookieJar
import urllib2
from urllib2 import urlopen


cj = CookieJar()
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
opener.addheaders = [("User-agent", "Mozilla/5.0")]

key_word = 'obama'

def main():
	try:
		source_code = opener.open("https://twitter.com/search/realtime?q=" + key_word + "&src=hash").read()
		split_source = re.findall(r'<p class="TweetTextSize  js-tweet-text tweet-text" lang="en" data-aria-label-part="0">(.*?)</p>', source_code)
		print len(split_source)
		for item in split_source:
			print item
			exit()
	except Exception, e:
		print str(e)
		print "Error in main try."
		time.sleep(30)
		
main()