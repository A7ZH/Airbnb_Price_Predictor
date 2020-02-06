from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

options = webdriver.ChromeOptions()
options.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
options.add_argument('--headless')
options.add_argument('--window-size=1920,1080')
options.add_argument('--disable-extensions')
options.add_argument('--diable-gpu')
options.add_argument('--incognito')
driver = webdriver.Chrome(chrome_options=options)

websites = open('website.txt', 'r').readlines()
output = open('locations.txt', 'w+')
error = open('error_urls.txt', 'w+')

for website in websites:
    driver.get(website[1:-2])
    
    try:
      WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH,
                  "//button[@class='optanon-allow-all accept-cookies-button']")))
      WebDriverWait(driver,10).until(EC.visibility_of_element_located((By.XPATH,
                  "//button[@class='optanon-allow-all accept-cookies-button']")))
      WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,
                  "//button[@class='optanon-allow-all accept-cookies-button']"))).click()
      WebDriverWait(driver, 10).until_not(EC.element_to_be_clickable((By.XPATH,
                  "//button[@class='optanon-allow-all accept-cookies-button']")))
      WebDriverWait(driver,10).until_not(EC.visibility_of_element_located((By.XPATH,
                  "//button[@class='optanon-allow-all accept-cookies-button']")))
    except:
      None

    try: 
      WebDriverWait(driver,60).until(EC.presence_of_element_located((By.XPATH,
                                     "//a[@title='Open this area in Google Maps (opens a new window)']")))
      WebDriverWait(driver,60).until(EC.visibility_of_element_located((By.XPATH,
                                     "//a[@title='Open this area in Google Maps (opens a new window)']")))
      url = WebDriverWait(driver,60).until(EC.element_to_be_clickable((By.XPATH,
                                     "//a[@title='Open this area in Google Maps (opens a new window)']"))) \
                                    .get_attribute('href')
      coord = (url.split('ll=')[1].split(',')[0], url.split(',')[1].split('&')[0])
      print(str(coord), file=output)
      print(str(coord))
    except:
      print(str((None, None)), file=output)
      print(str((None, None)))
      print(website[1:-2], file=error)
      
driver.close()
