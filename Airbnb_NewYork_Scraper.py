from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Add options for chrome broswer.
options = webdriver.ChromeOptions()
options.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
options.add_argument('--headless')
options.add_argument('--window-size=1920,1080')
options.add_argument('--disable-extensions')
options.add_argument('--disable-gpu')
options.add_argument('--incognito')

# Run Chrome browser 
driver = webdriver.Chrome(chrome_options=options)

# Create a file to save all the New York City listings scraped from Airbnb 
output = open("Airbnb_NewYork_Scraper_Output2.txt", "w+")

# Give each key word a crawling iteration
key_words = ["New York City, NY", 
             "Manhattan, New York, NY", "Brooklyn, New York, NY", "Queens, New York, NY",
             "Bronx, New York, NY", "Staten Island, New York, NY"]
iteration=1

for key_word in key_words:
  # navigate to "www.airbnb.ca" webpage.
  driver.get('https://www.airbnb.ca')

  try:
    # Wait for the "Accept Cookies" pop-up to present in DOM, become visible and clickable, and click "OK" 
    #  to dismiss it. After that, wait for the pop-up to become unclickable and invisible.
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                  "//button[@class='optanon-allow-all accept-cookies-button']"))) 
    WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH,
                                  "//button[@class='optanon-allow-all accept-cookies-button']"))) 
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,
                                  "//button[@class='optanon-allow-all accept-cookies-button']"))).click()
    WebDriverWait(driver, 10).until_not(EC.element_to_be_clickable((By.XPATH,
                                  "//button[@class='optanon-allow-all accept-cookies-button']")))
    WebDriverWait(driver, 10).until_not(EC.visibility_of_element_located((By.XPATH,
                                  "//button[@class='optanon-allow-all accept-cookies-button']"))) 
    # Waiting for the pop-up should have earned the webpage enough time to fully load.
  except:
    # If the "Accept Cookies" pop-up does not present in DOM, or become visible and clickable, log it
    #   and move on.
    print("LOG: Accept Cookie Pop-up No Show")
  
  if iteration==1:
    # Click on the currency button in the top-right navigation bar.
    driver.find_element_by_xpath("//header[@role='banner']") \
          .find_element_by_tag_name('nav') \
          .find_elements_by_tag_name('li')[1] \
          .find_element_by_tag_name('button') \
          .click()

    # Wait for the $USD option in the currency selection menu to present in DOM, become visible and clickable, 
    #   and click to choose it. After that, wait for the selection menu to become invisible and unpresent.
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,
                                    "//ul[@class='_19s389u']/li[39]")))
    WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.XPATH,
                                    "//ul[@class='_19s389u']/li[39]")))
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH,
                                    "//ul[@class='_19s389u']/li[39]"))).click()
    WebDriverWait(driver, 10).until_not(EC.visibility_of_element_located((By.XPATH, 
                                    "//div[@aria-label='Currencies']")))
    WebDriverWait(driver, 10).until_not(EC.presence_of_element_located((By.XPATH, 
                                    "//div[@aria-label='Currencies']")))

  # Fill the location field in the search form
  driver.find_element_by_xpath("//input[@placeholder='Anywhere']").send_keys(key_word) 
  driver.find_element_by_xpath("//input[@placeholder='Anywhere']").send_keys(Keys.RETURN)

  # Click "Search" button in the search form
  driver.find_element_by_xpath("//button[@type='submit']").click()

  # Wait for the "Stays Experience Adventures" selection page to load, present in DOM, become visible
  #   clickable, and click "Stays" to proceed.
  WebDriverWait(driver,30).until(EC.presence_of_element_located((By.XPATH,"//div[@title='Stays']")))
  WebDriverWait(driver,30).until(EC.visibility_of_element_located((By.XPATH,"//div[@title='Stays']")))
  WebDriverWait(driver,30).until(EC.element_to_be_clickable((By.XPATH,"//div[@title='Stays']"))).click()

  while True: # Do the following for each listing page
    # Wait for the listings in the listing page to load, present in DOM, become visible and clickable.
    WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH,
                "//div[@class='_8ssblpx']/descendant::div[@class='_1i2fr3fi']")))
    WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.XPATH,
                "//div[@class='_8ssblpx']/descendant::div[@class='_1i2fr3fi']")))
    WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH,
                "//div[@class='_8ssblpx']/descendant::div[@class='_1i2fr3fi']")))
    # Acquire all the listing element in the page
    listings = driver.find_elements_by_xpath("//div[@class='_8ssblpx']")
    l = len(listings)
    for i in range(l):
      info1 = listings[i].text.replace('\n', ' ; ')   # listing text info
      info2 = listings[i].find_element_by_xpath("descendant::meta[@itemprop='name']") \
                         .get_attribute('content') \
                         .split(" - undefined - ")[1] # listing title attribute, some including neighbourhood
      info3 = listings[i].find_element_by_xpath("descendant::a[1]") \
                         .get_attribute('href')       # listing webpage link attribute, including listing ID#
      #info4 = listings[i].find_element_by_xpath("descendant::div[@class='_1i2fr3fi']") \
      #                   .get_attribute('style') \
      #                   .split("url(\"")[1][:-3]     # listing cover picture link attribute
      print(info1 + "\n" + info2 + "\n" + info3 + "\n", file=output)
      print(info1 + "\n" + info2 + "\n" + info3 + "\n")
    # Wait for the page number navigation bar at the bottom of the page to load, present in DOM, become
    #  visible and clickable
    WebDriverWait(driver,30).until(EC.presence_of_element_located((By.XPATH, 
          "//nav[@class='_w9uwpe'][@data-id='SearchResultsPagination']/descendant::li[@class='_ycd2pg']")))
    WebDriverWait(driver,30).until(EC.visibility_of_element_located((By.XPATH, 
          "//nav[@class='_w9uwpe'][@data-id='SearchResultsPagination']/descendant::li[@class='_ycd2pg']")))
    WebDriverWait(driver,30).until(EC.element_to_be_clickable((By.XPATH, 
          "//nav[@class='_w9uwpe'][@data-id='SearchResultsPagination']/descendant::li[@class='_ycd2pg']")))
    pages = driver.find_elements_by_xpath(
          "//nav[@class='_w9uwpe'][@data-id='SearchResultsPagination']/descendant::li[@class='_ycd2pg']")
    last_page_num = int(pages[-1].get_attribute('data-id').split('-')[1])
    p_ind = 1
    for p in pages:
      label = p.find_element_by_xpath("child::*").get_attribute('aria-label')
      if (len(label.split(', '))>1) and (label.split(', ')[1] == "current page"): 
        current_page_num = int(label.split(', ')[0].split(' ')[1])
        break
      p_ind+=1
    if(current_page_num==last_page_num): break
    else: 
      driver.get(pages[p_ind].find_element_by_xpath("child::*").get_attribute('href')) 
      driver.implicitly_wait(3)
  iteration+=1
driver.close()
