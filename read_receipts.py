### Requirement: Read the sample receipts and extract barcode data, total of the bill, company logo and all other readable text.

import os, io
import pyzbar.pyzbar as pyzbar
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdf2image
from PIL import ImageOps, ImageEnhance
import pytesseract as ptess
import re
from google.cloud import vision
from google.cloud.vision import types

# Setup
ptess.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\\Users\\jquirk4\\OneDrive - DXC Production\\DXC\\Data-Science-Projects\\Project 1 - receipt reading\\GCP_Vision_Credentials.json'

vison_client = vision.ImageAnnotatorClient()

# Class
class ReceiptReader:

    def __init__(self, url, highlight=False, greyscale=True):
        # Initialise Parameters
        self._HIGHLIGHT = highlight
        self._GRAYSCALE = greyscale
        self.IMAGE_URL = url
        self.barcode_count = 0
        self.barcodes = {}  # {'type1':{'count':0,'data':[]}}
        self.detected_objects = []  # [{'label':'','poly':[{'x':0,'y':0},{'x':0,'y':0}...],'data':''},...]

        # Initialise receipt images
        self.get_image_from_url(url)

        # Image pre-processing
        self.image_preprocessing()


    def get_image_from_url(self,url):
        # Returns python list of CV2 images (receipt may contain multiple pages) - numpy arrays with third dimension in BGR format.
        images = []
        if url[-3:] == 'pdf':
            print('Receipt document type: PDF')
            pages = pdf2image.convert_from_path(url,dpi=300)  # returns each page of the PDF as a PIL image.
            self.PIL_images = pages
            for page in pages:
                page = page.convert('RGB')
                open_cv_image = np.array(page)
                # Convert RGB to BGR (open cv image standard)
                images.append(open_cv_image[:,:,::-1].copy())   # ::-1 just reverses the order
        else:
            print('Receipt document type: %s' % url.split('.')[-1].upper())
            images.append(cv2.imread(url))
        self.raw_images = images

        if len(self.raw_images) > 1:
            self.raw_images.pop(1)   # TODO: Delete

    
    def image_preprocessing(self):
        if self._GRAYSCALE:
            self.processed_images = []
            for img in self.raw_images:
                img_proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # ReceiptReader.plot_hist(img_proc)

                # BINARISATION
                for row_i, row in enumerate(img_proc):
                    for p_i, p in enumerate(row):
                        if p < int(253):
                            img_proc[row_i][p_i] = 0
                        else:
                            img_proc[row_i][p_i] = 255

                # ReceiptReader.plot_hist(img_proc)

                self.processed_images.append(img_proc)
        else:
            self.processed_images = self.raw_images.copy()


    def read_barcode(self):
        for img in self.processed_images:
            decodedObjects = pyzbar.decode(img)

            for obj in decodedObjects:
                self.barcode_count += 1
                if obj.type in self.barcodes.keys():
                    self.barcodes[obj.type]['count'] += 1
                    self.barcodes[obj.type]['data'].append(obj.data)
                else:
                    self.barcodes[obj.type] = {'count':1,'data':[obj.data]}

                self.detected_objects.append({
                    'label':'barcode',
                    'poly':[
                        {'x':p.x,'y':p.y} for p in obj.polygon
                    ],
                    'details':obj.data
                })

                print('Type: ', obj.type)
                print('Data: ', obj.data,'\n')
            
            if self._HIGHLIGHT:
                h_img = self.raw_images[self.processed_images.index(img)]
                imgArr = np.array(h_img)
                self.highlight_barcode(imgArr, decodedObjects)


    def highlight_barcode(self, img, objs): # TODO: Create 'found objects' to contain objects from the image to then later highlight at once
        for obj in objs:
            points = obj.polygon

            if len(points) > 4:
                hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                hull = list(map(tuple, np.squeeze(hull)))
            else:
                hull = points

            n = len(hull)

            for j in range(0,n):
                cv2.line(img, hull[j], hull[(j+1)%n], (255,0,0),3)
        
        ReceiptReader.show_image(img)


    def overlay_objects(self):
        '''Add detected objects to the image'''

        for obj in detected_objects:
            pass


    def read_text(self):
        self.text_list = []
        self.bill_total = None
        self.bill_date = None
        for img in self.processed_images:
            text = ptess.image_to_string(img)
            self.text_list.append(text)

            print(text[:100])

            # Extract bill total
            total_search = re.search(r'^Total.{0,3}?([0-9]*\.[0-9]*)$',text,flags=re.MULTILINE|re.IGNORECASE)
            if total_search:
                print(total_search)
                self.bill_total = total_search.group(1)
            elif self.bill_total is None:
                # print(text[:100])
                pass

            # Extract date
            date_search = re.search(r'([0-9]{1,2}(-|/)[0-9]{1,2}(-|/)[0-9]{2,4}|[0-9]{2,4}(-|/)[0-9]{1,2}(-|/)[0-9]{1,2})',text,flags=re.MULTILINE|re.IGNORECASE)
            if date_search:
                print(date_search)
                self.bill_date = date_search.group(0)


    def detect_logos(self):
        # Convert PIL to byte array
        imgByteArr = io.BytesIO()
        self.PIL_images[0].save(imgByteArr, format=self.PIL_images[0].format)   # PPM format
        imgByteArr = imgByteArr.getvalue()

        # with open(self.IMAGE_URL,'rb') as imfile:
        #     imgBytes = imfile.read()
        #     print(imgBytes)

        image = vision.types.Image(content=imgByteArr)
        response = vison_client.logo_detection(image=image)
        # response = vison_client.text_detection(image=image)
        return response



    @staticmethod
    def show_image(img,windowName='Results',scale=2.5):
        # Takes img as numpy array
        rows = img.shape[0]
        cols = img.shape[1]
        cv2.namedWindow(windowName,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, int(cols/scale), int(rows/scale))
        cv2.imshow(windowName, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    @staticmethod
    def plot_hist(img):
        plt.hist(img)
        plt.show()


if __name__ == '__main__':
    dir_ = os.path.dirname(__file__)

    # img_name = 'walgreens-20191207.pdf'
    img_name = 'cocacola-20191221_001.pdf'
    # img_name = 'test.png'
    sample_dir = dir_ + '/sample1/sample1/'

    receipts = os.listdir(sample_dir)

    for rec in [receipts[receipts.index(img_name)]]:
        url = sample_dir + rec
        print('Receipt name: %s' % rec)
        reader = ReceiptReader(url)


        reader.read_barcode()
        
        # reader.read_text()
        # print('Total: %s' % reader.bill_total)
        # print('Date: %s' % reader.bill_date)
        # print(reader.barcodes)

        # reader.show_image(reader.processed_images[0])

        response = reader.detect_logos()
        print(response)

        

    # print(reader.barcode_count)
    # print(reader.barcodes)



