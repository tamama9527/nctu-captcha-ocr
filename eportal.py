
import requests
import io
from PIL import Image
from PIL import ImageEnhance
import numpy as np
import cv2
from bs4 import BeautifulSoup
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import csv
import time
from keras.models import load_model
model = load_model('1.h5')

LETTERSTR = "123456789abcdefghijkmonpqrstuvwxyz"


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


code_page = 'https://sso.nutc.edu.tw/ePortal/Validation_Code.aspx'
url = 'https://sso.nutc.edu.tw/ePortal/Default.aspx'
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:40.0) Gecko/20100101 Firefox/40.1' }
fail = 0.0
success = 0.0
total = 0
for j in range(5000):
	s = requests.session()
	try:
		r = s.get(url, headers=header, verify=False)
	except:
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
	r = s.get(code_page, headers=header, verify=False)
	image = Image.open((io.BytesIO(r.content)))
	enh_con = ImageEnhance.Contrast(image)
	image_contrasted = enh_con.enhance(1.7)
	dst = cv2.fastNlMeansDenoisingColored(np.array(
		gray(image_contrasted).convert('RGB')), None, 40, 40, 7, 30)
	ret, thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY)
	im = Image.fromarray(thresh)
	prediction = model.predict(np.expand_dims(np.stack([np.array(im.convert('1')) / 255.0]), axis=-1))
	answer = ''
	for predict in prediction:
		answer += LETTERSTR[np.argmax(predict[0])]
	data['ctl00$ContentPlaceHolder1$ValidationCode'] = answer
	try:
		r = s.post(url, headers=header, data=data, verify=False)
	except:
		continue
	total = total + 1
	if r.url == 'https://sso.nutc.edu.tw/ePortal/myarea/MyArea.aspx':
		success = success + 1
	else:
		fail = fail + 1

	try:
		s.get('https://sso.nutc.edu.tw/ePortal/Logout.aspx',headers=header)
	except:
		continue
	print('總次數:'+str(total)+'成功機率:{}'.format((success/(success+fail))*100,'2.2f'))
	

