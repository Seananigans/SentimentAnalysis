import matplotlib.pyplot as plt
import numpy as np
import urllib
import matplotlib.dates as mdates

# x = [1,2,3]
# y = [5,7,4]
# 
# x2 = [4,5,6]
# y2 = [10, 13, 17]
# 
# plt.bar(x,y, label="First", color="r")
# plt.bar(x2,y2, label="Second", color="c")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Some Graph")
# plt.legend()
# plt.show()
# 
# days = [1,2,3,4,5]
# 
# sleeping = [7,8,6,11,7]
# eating =   [2,3,4,3,2]
# working =  [7,8,7,2,2]
# playing =  [8,5,7,8,13]
# 
# plt.plot([],[], color='b', label="Playing", linewidth=5)
# plt.plot([],[], color='g', label="Working", linewidth=5)
# plt.plot([],[], color='k', label="Eating", linewidth=5)
# plt.plot([],[], color='m', label="Sleeping", linewidth=5)
# 
# plt.stackplot(days, sleeping, eating, working, playing, colors = ["m","k","g","b"])
# 
# plt.xlabel("X label")
# plt.ylabel("Y label")
# plt.ylim((0,24))
# plt.title("stackplot")
# plt.legend()
# plt.show()
# 
# 
# slices = [7,2,2,13]
# activities = ['sleeping','eating','working','playing']
# cols = ['c','m','r','b']
# 
# plt.pie(slices, labels = activities, colors = cols, shadow=True, startangle=90, explode=(0,0.2,0,0), autopct="%1.1f%%")
# 
# plt.title("Pie Chart\n...mmm pie.")
# plt.show()

def bytespdate2num(fmt, encoding='utf-8'):
	strconverter = mdates.strpdate2num(fmt)
	def bytesconverter(b):
		s = b.decode(encoding)
		return strconverter(s)
	return bytesconverter

def graph_data(stock):

	fig = plt.figure()
	ax1 = plt.subplot2grid((1,1), (0,0))
	
	stock_price_url = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=10y/csv'
	
	source_code = urllib.urlopen(stock_price_url).read().decode()
	
	stock_data = []
	split_source = source_code.split('\n')
	
	for line in split_source:
		split_line = line.split(',')
		if len(split_line)==6:
			if 'values' not in line:
				stock_data.append(line)
	
	date, closep, highp, lowp, openp, volume = np.loadtxt(	stock_data, 
															delimiter=',', 
															unpack=True, 
															converters={0:bytespdate2num('%Y%m%d')})
	
	ax1.plot_date(date,closep, '-', label="loaded from file")
	for label in ax1.xaxis.get_ticklabels():
		label.set_rotation(45)
	ax1.grid(True)
	
	plt.xlabel("Date")
	plt.ylabel("Price")
	plt.title("Prices of {} over time".format(stock))
	plt.legend()
	
	plt.subplots_adjust(left=0.1, bottom=0.20, right=0.94, top=0.92, wspace=0.2, hspace=0)
	plt.show()


graph_data("TSLA")










