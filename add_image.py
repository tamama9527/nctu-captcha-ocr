#!/usr/bin/env python
# coding:utf-8
import time
import requests
from hashlib import md5
import io
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import cv2
from bs4 import BeautifulSoup
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import csv
from keras.models import load_model
model = load_model('v2-10820.h5')

def gray(img):
    image = img
 # 去噪 二值化
    image = image.convert('RGBA')
    pixdata = image.load()
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            if pixdata[x, y][0] < 90:
                pixdata[x, y] = (0, 0, 0, 255)
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            if pixdata[x, y][1] < 136:
                pixdata[x, y] = (0, 0, 0, 255)
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            if pixdata[x, y][2] > 0:
                pixdata[x, y] = (255, 255, 255, 255)
    return image

LETTERSTR = "123456789abcdefghijkmonpqrstuvwxyz"
num = 2410

if __name__ == '__main__':

	code_page = 'https://sso.nutc.edu.tw/ePortal/Validation_Code.aspx'
	url = 'https://sso.nutc.edu.tw/ePortal/Default.aspx'
	header = {
		'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1'}
	for j in range(1200):
		try:
			s = requests.session()
			r = s.get(url, headers=header, verify=False)
		except:
			time.sleep(1)
			continue
		soup = BeautifulSoup(r.text, "lxml")
		data = {
			'ctl00$ContentPlaceHolder1$Account': 'account',
			'ctl00$ContentPlaceHolder1$Login.x': '25',
			'ctl00$ContentPlaceHolder1$Login.y': '37',
			'ctl00$ContentPlaceHolder1$Password': 'password',
			'ctl00$ContentPlaceHolder1$ValidationCode': ''
		}
		for i in soup.findAll('input', {'type': 'hidden', 'value': True}):
			data[i['name']] = i['value']
		try:
			r = s.get(code_page, headers=header, verify=False)
		except:
			time.sleep(1)
			continue
		image = Image.open((io.BytesIO(r.content)))
		enh_con = ImageEnhance.Contrast(image)
		image_contrasted = enh_con.enhance(1.7)
		dst = cv2.fastNlMeansDenoisingColored(np.array(
			gray(image_contrasted).convert('RGB')), None, 40, 40, 7, 30)
		ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
		prediction = model.predict(np.stack([thresh / 255.0]))

		answer = ''
		for predict in prediction:
			answer += LETTERSTR[np.argmax(predict[0])]
		data['ctl00$ContentPlaceHolder1$ValidationCode'] = answer
		try:
			r = s.post(url, headers=header, data=data, verify=False)
		except:
			time.sleep(1)
			continue
		cv2.imwrite('black/{}.png'.format(j + num), thresh)
		if r.url != 'https://sso.nutc.edu.tw/ePortal/myarea/MyArea.aspx':
			print(str(j+num)+':'+answer)
			answer = '*****'
		with open('label2.csv', 'a+', newline='') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow([j + num, answer])
		try:
			s.get('https://sso.nutc.edu.tw/ePortal/Logout.aspx',headers=header)
		except:
			time.sleep(1)
			continue
