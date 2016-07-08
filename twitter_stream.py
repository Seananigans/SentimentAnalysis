from tweepy import Stream, OAuthHandler
from tweepy.streaming import StreamListener
import json, time

ckey = "VcfqmI3RzvDoyQpnL9lCyH87R"
csecret = "DzCdRto1AxXaHrpBfTrBAwCnrqY95XgDRrm6Uk7SUkZ1hY8Hgt"
atoken = "1938505076-qOOyl37ZQdD8NZyj9JUXcCFW74cmb6jmcHkpoog"
asecret = "uxgl0Q6EJQXDqKzSeuxgoA2Y2ggrtRvQkTSKO0eLQeEON"

class Listener(StreamListener):
	
	def on_data(self, data):
		try:
			# Get tweet by splitting data
# 			tweet = data.split(',"text":"')[1]
# 			tweet = tweet.split('","source"')[0]

			# Get tweet with JSON library
			all_data = json.loads(data)
			tweet = all_data['text'].encode('utf-8')
			
			#save data
			save_this = str(time.time())+"::::"+tweet
			saveFile = open("twitDB.csv","a")
			saveFile.write(save_this)
			saveFile.write("\n")
			saveFile.close()
			
			
			return True
		except BaseException, e:
			print "Failed on_data,",str(e)
			time.sleep(5)
		
	def on_error(self, status):
		print status


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

current_time = time.time()
twitterStream = Stream(auth, Listener())
twitterStream.filter(track=["obama"])