import requests
from bs4 import BeautifulSoup as bs
import re
import csv 

b = 1
with open('data_set.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["ArticleID", "news"])
	web_url = "https://scroll.in/article/"
	for j in range(5000):
		raw_data = requests.get(web_url+str(j+953000), "lxml")
		soup = bs(raw_data.content, "lxml")
		m = re.search(r'<p>.*</p>',str(soup), re.MULTILINE)
		if m is not None:
			news = ''
			paras = soup.find_all('p')
			for i in range(len(paras)):
				a1 = paras[i].get_text()
				news = news + a1
			if len(news) != 0:
				writer.writerow([b, news])
				print(b)
				b= b+1
			
			
			
			
