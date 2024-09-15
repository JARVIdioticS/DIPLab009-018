import math
from os import name
import random
from PIL import Image
import numpy as np
import cmath
import sys
from FrequencyDomainManager import FrequencyDomainManager

class ImageManager:
    width = None
    height = None
    bitDepth = None

    img = None
    data = None
    original = None

    def read(self, fileName):
        global img 
        global data 
        global original 
        global width 
        global height 
        global bitDepth 
        img = Image.open(fileName)
        data = np.array(img)
        original = np.copy(data)
        width = data.shape[1]
        height = data.shape[0]

        mode_to_bpp = {"1":1,"L":8,"P":8,"RGB":24,"RGBA":32,"CMYK":32,"YCbCr":24,"LAB":24,"HSV":24,"I":32,"F":32}
        bitDepth = mode_to_bpp[img.mode]

        print("Image %s with %s x %s pixels (%s bits per pixels) has been read!" % (img.filename, width, height, bitDepth))

    def write(self, fileName):
        global img 
        img = Image.fromarray(data)
        try:
            img.save(fileName)
        except:
            print("Write file error")
        else:
            print("Image %s has been written!" %(fileName))

    def getFrequencyDomain(self):
        self.convertToGray()
        fft = FrequencyDomainManager(self)
        self.restoreToOriginal()
        return fft

    def convertToRed(self):
        global data
        for y in range(height):
            for x in range(width):
                data[x, y, 1] = 0
                data[x, y, 2] = 0

    def convertToGreen(self):
        global data
        for y in range(height):
            for x in range(width):
                data[x, y, 0] = 0
                data[x, y, 2] = 0

    def convertToBlue(self):
        global data
        for y in range(height):
            for x in range(width):
                data[x, y, 0] = 0
                data[x, y, 1] = 0

    def convertToGray(self):
        global data
        for y in range(height):
            for x in range(width):
                # matlab's (NTSC/PAL) implementation:
                R, G, B = data[x, y, 0], data[x, y, 1], data[x, y, 2]
                gray = 0.2989 * R + 0.5870 * G + 0.1140 * B # and somehow the magic happened
                data[x, y, 0], data[x, y, 1], data[x, y, 2] = gray, gray, gray

    def restoreToOriginal(self):
        global data
        width = original.shape[0]
        height = original.shape[1]
        data = np.zeros([width, height, 3])
        data = np.copy(original)

    def adjustBrightness(self, brightness):
        global data
        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]

                r = r + brightness
                r = 255 if r > 255 else r
                r = 0 if r < 0 else r

                g = g + brightness
                g = 255 if g > 255 else g
                g = 0 if g < 0 else g

                b = b + brightness
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b

                data[x, y, 0] = r
                data[x, y, 1] = g
                data[x, y, 2] = b

    def invert(self):
        global data
        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]

                r = 255 - r
                g = 255 - g
                b = 255 - b

                data[x, y, 0] = r
                data[x, y, 1] = g
                data[x, y, 2] = b

    def powerLaw(self, constant, gamma):
        global data
        for y in range(height):
            for x in range(width):
                r = data[x, y, 0] / 255.0
                g = data[x, y, 1] / 255.0
                b = data[x, y, 2] / 255.0

                r = (int)(255 * (constant * (math.pow(r, gamma))))
                r = 255 if r > 255 else r
                r = 0 if r < 0 else r

                g = (int)(255 * (constant * (math.pow(g, gamma))))
                g = 255 if g > 255 else g
                g = 0 if g < 0 else g

                b = (int)(255 * (constant * (math.pow(b, gamma))))
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b

                data[x, y, 0] = r
                data[x, y, 1] = g
                data[x, y, 2] = b
        return

    def getGrayscaleHistogram(self):
        self.convertToGray()

        histogram = np.array([0] * 256)

        for y in range(height):
            for x in range(width):
                histogram[data[x, y, 0]] += 1

        self.restoreToOriginal()
        return histogram
            
    def writeHistogramToCSV(self, histogram, fileName):
        histogram.tofile(fileName,sep=',',format='%s')

    def getContrast(self):
        contrast = 0.0
        histogram = self.getGrayscaleHistogram()
        avgIntensity = 0.0
        pixelNum = width * height

        for i in range(len(histogram)):
            avgIntensity += histogram[i] * i

        avgIntensity /= pixelNum

        for y in range(height):
            for x in range(width):
                contrast += (data[x, y, 0] - avgIntensity) ** 2

        contrast = (contrast / pixelNum) ** 0.5

        return contrast
                
    def adjustContrast(self, contrast):
        global data
        currentContrast = self.getContrast()
        histogram = self.getGrayscaleHistogram()
        avgIntensity = 0.0
        pixelNum = width * height
        for i in range(len(histogram)):
            avgIntensity += histogram[i] * i

        avgIntensity /= pixelNum

        min = avgIntensity - currentContrast
        max = avgIntensity + currentContrast

        newMin = avgIntensity - currentContrast - contrast / 2
        newMax = avgIntensity + currentContrast + contrast / 2

        newMin = 0 if newMin < 0 else newMin
        newMax = 0 if newMax < 0 else newMax
        newMin = 255 if newMin > 255 else newMin
        newMax = 255 if newMax > 255 else newMax

        if (newMin > newMax):
            temp = newMax
            newMax = newMin
            newMin = temp

        contrastFactor = (newMax - newMin) / (max - min)

        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]
                contrast += (data[x, y, 0] - avgIntensity) ** 2

                r = (int)((r - min) * contrastFactor + newMin)
                r = 255 if r > 255 else r
                r = 0 if r < 0 else r

                g = (int)((g - min) * contrastFactor + newMin)
                g = 255 if g > 255 else g
                g = 0 if g < 0 else g

                b = (int)((b - min) * contrastFactor + newMin)
                b = 255 if b > 255 else b
                b = 0 if b < 0 else b

                data[x, y, 0] = r
                data[x, y, 1] = g
                data[x, y, 2] = b

    def averagingFilter(self, size):
        global data
        if (size % 2 == 0):
            print("Size Invalid: must be odd number!")
            return
        
        data_zeropaded = np.zeros([width + int(size/2) * 2, height + int(size/2) * 2, 3])
        data_zeropaded[int(size/2):width + int(size/2), int(size/2):height + int(size/2), :] = data

        for y in range(int(size/2), int(size/2) + height):
            for x in range(int(size/2), int(size/2) + width):

                subData = data_zeropaded[x - int(size/2):x + int(size/2) + 1, y - int(size/2):y + int(size/2) + 1, :]

                avgRed = np.mean(subData[:,:,0:1])
                avgGreen = np.mean(subData[:,:,1:2])
                avgBlue = np.mean(subData[:,:,2:3])

                avgRed = 255 if avgRed > 255 else avgRed
                avgRed = 0 if avgRed < 0 else avgRed

                avgGreen = 255 if avgGreen > 255 else avgGreen
                avgGreen = 0 if avgGreen < 0 else avgGreen

                avgBlue = 255 if avgBlue > 255 else avgBlue
                avgBlue = 0 if avgBlue < 0 else avgBlue

                data[x - int(size/2), y - int(size/2), 0] = avgRed
                data[x - int(size/2), y - int(size/2), 1] = avgGreen
                data[x - int(size/2), y - int(size/2), 2] = avgBlue

    def unsharpFilter(self, size, k):
        global data
        if (size % 2 == 0):
            print("Size Invalid: must be odd number!")
            return
        
        tmpimg = np.copy(data)
        self.averagingFilter(size)
        
        data = np.clip(tmpimg + k * (tmpimg - data), 0, 255)

    def gaussianBlurFilter(self, size, sigma):
        global data

        if size % 2 == 0:
            print("Size Invalid: must be odd number!")
            return

        k = int(size // 2)
        x, y = np.mgrid[-k:k+1, -k:k+1]
        gaussian_kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        gaussian_kernel /= gaussian_kernel.sum()

        data_zeropaded = np.zeros([width + k * 2, height + k * 2, 3])
        data_zeropaded[k:width + k, k:height + k, :] = data

        for y in range(k, k + height):
            for x in range(k, k + width):
                subData = data_zeropaded[x - k:x + k + 1, y - k:y + k + 1, :]
                
                red = np.sum(subData[:, :, 0] * gaussian_kernel)
                green = np.sum(subData[:, :, 1] * gaussian_kernel)
                blue = np.sum(subData[:, :, 2] * gaussian_kernel)

                data[x - k, y - k, 0] = min(max(red, 0), 255)
                data[x - k, y - k, 1] = min(max(green, 0), 255)
                data[x - k, y - k, 2] = min(max(blue, 0), 255)

    def medianFilter(self, size):
        global data

        if size % 2 == 0:
            print("Size Invalid: must be odd number!")
            return

        k = int(size // 2)
        data_zeropaded = np.zeros([width + k * 2, height + k * 2, 3])
        data_zeropaded[k:width + k, k:height + k, :] = data

        for y in range(k, k + height):
            for x in range(k, k + width):
                subData = data_zeropaded[x - k:x + k + 1, y - k:y + k + 1, :]

                medianRed = np.median(subData[:, :, 0])
                medianGreen = np.median(subData[:, :, 1])
                medianBlue = np.median(subData[:, :, 2])

                data[x - k, y - k, 0] = medianRed
                data[x - k, y - k, 1] = medianGreen
                data[x - k, y - k, 2] = medianBlue

    def laplacianFilter(self):
        global data
        laplacian_kernel = np.array([[0, -1, 0],
                                    [-1, 4, -1],
                                    [0, -1, 0]])

        data_zeropaded = np.zeros([width + 2, height + 2, 3])
        data_zeropaded[1:width + 1, 1:height + 1, :] = data

        for y in range(1, height + 1):
            for x in range(1, width + 1):
                subData = data_zeropaded[x - 1:x + 2, y - 1:y + 2, :]

                laplacianRed = np.sum(subData[:, :, 0] * laplacian_kernel)
                laplacianGreen = np.sum(subData[:, :, 1] * laplacian_kernel)
                laplacianBlue = np.sum(subData[:, :, 2] * laplacian_kernel)

                red = data[x - 1, y - 1, 0] - laplacianRed
                green = data[x - 1, y - 1, 1] - laplacianGreen
                blue = data[x - 1, y - 1, 2] - laplacianBlue

                data[x - 1, y - 1, 0] = min(max(red, 0), 255)
                data[x - 1, y - 1, 1] = min(max(green, 0), 255)
                data[x - 1, y - 1, 2] = min(max(blue, 0), 255)

    def addSaltNoise(self, percent):
        global data
        noOfPX = height * width
        noiseAdded = (int)(percent * noOfPX)
        whiteColor = 255
        for i in range(noiseAdded):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            data[x, y, 0] = whiteColor
            data[x, y, 1] = whiteColor
            data[x, y, 2] = whiteColor

    def addPepperNoise(self, percent):
        global data
        noOfPX = height * width
        noiseAdded = (int)(percent * noOfPX)
        blackColor = 0
        for i in range(noiseAdded):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            data[x, y, 0] = blackColor
            data[x, y, 1] = blackColor
            data[x, y, 2] = blackColor

    def addSaltAndPepperNoise(self, percent):
        self.addSaltNoise(percent)
        self.addPepperNoise(percent)

    def contraharmonicFilter(self, size, Q):
        global data
        if (size % 2 == 0):
            print("Size Invalid: must be odd number!")
            return
        
        data_temp = np.zeros([width, height, 3])
        data_temp = data.copy()

        for y in range(height):
            for x in range(width):

                sumRedAbove = 0
                sumGreenAbove = 0
                sumBlueAbove = 0
                sumRedBelow = 0
                sumGreenBelow = 0
                sumBlueBelow = 0

                subData = data_temp[x - int(size/2):x + int(size/2) + 1, y - int(size/2):y + int(size/2) + 1, :].copy()
                subData = subData ** (Q + 1)
                sumRedAbove = np.sum(subData[:,:,0:1], axis=None)
                sumGreenAbove = np.sum(subData[:,:,1:2], axis=None)
                sumBlueAbove = np.sum(subData[:,:,2:3], axis=None)

                subData = data_temp[x - int(size/2):x + int(size/2) + 1, y - int(size/2):y + int(size/2) + 1, :].copy()
                subData = subData ** Q
                sumRedBelow = np.sum(subData[:,:,0:1], axis=None)
                sumGreenBelow = np.sum(subData[:,:,1:2], axis=None)
                sumBlueBelow = np.sum(subData[:,:,2:3], axis=None)

                if (sumRedBelow != 0): sumRedAbove /= sumRedBelow
                sumRedAbove = 255 if sumRedAbove > 255 else sumRedAbove
                sumRedAbove = 0 if sumRedAbove < 0 else sumRedAbove
                if (math.isnan(sumRedAbove)): sumRedAbove = 0

                if (sumGreenBelow != 0): sumGreenAbove /= sumGreenBelow
                sumGreenAbove = 255 if sumGreenAbove > 255 else sumGreenAbove
                sumGreenAbove = 0 if sumGreenAbove < 0 else sumGreenAbove
                if (math.isnan(sumGreenAbove)): sumGreenAbove = 0

                if (sumBlueBelow != 0): sumBlueAbove /= sumBlueBelow
                sumBlueAbove = 255 if sumBlueAbove > 255 else sumBlueAbove
                sumBlueAbove = 0 if sumBlueAbove < 0 else sumBlueAbove
                if (math.isnan(sumBlueAbove)): sumBlueAbove = 0

                data[x, y, 0] = sumRedAbove
                data[x, y, 1] = sumGreenAbove
                data[x, y, 2] = sumBlueAbove

    def alphaTrimmedFilter(self, size, d):
        global data

        # ตรวจสอบขนาดฟิลเตอร์
        if size % 2 == 0:
            print("Size Invalid: must be odd number!")
            return

        if d >= size * size:
            print("Invalid d: too large for given size!")
            return

        pad_size = size // 2
        data_zeropaded = np.zeros([width + 2 * pad_size, height + 2 * pad_size, 3])
        data_zeropaded[pad_size:width + pad_size, pad_size:height + pad_size, :] = data

        for y in range(height):
            for x in range(width):
                # ดึงค่าข้อมูลที่ต้องการ
                subData = data_zeropaded[x:x + size, y:y + size, :]

                # เรียงค่าพิกเซลในแต่ละช่องสี
                sortedRed = np.sort(subData[:, :, 0].flatten())
                sortedGreen = np.sort(subData[:, :, 1].flatten())
                sortedBlue = np.sort(subData[:, :, 2].flatten())

                # ตัดค่าที่ไม่ต้องการออกและคำนวณค่าเฉลี่ย
                r = np.mean(sortedRed[d//2 : -d//2])
                g = np.mean(sortedGreen[d//2 : -d//2])
                b = np.mean(sortedBlue[d//2 : -d//2])

                # ป้องกันไม่ให้ค่าเกินขอบเขต
                data[x, y, 0] = min(max(0, r), 255)
                data[x, y, 1] = min(max(0, g), 255)
                data[x, y, 2] = min(max(0, b), 255)

    def convertToRedBlue(self, cr, cg, cb):
        global data
        for y in range(height):
            for x in range(width):
                r = data[x, y, 0]
                g = data[x, y, 1]
                b = data[x, y, 2]

                if (r <= 10 and g <= 10 and b <= 10) or (r >= 245 and g >= 245 and b >= 245): continue

                r = min(255, r + cr)
                g = max(0, g + cg)
                b = min(255, b + cb)

                data[x, y, 0] = r
                data[x, y, 1] = g
                data[x, y, 2] = b

    def resizeNearestNeighbour(self, scaleX, scaleY):
        global data
        global width
        global height
        newWidth = (int)(round(width * scaleX))
        newHeight = (int)(round(height * scaleY))

        data_temp = np.zeros([width, height, 3])
        data_temp = data.copy()

        data = np.resize(data, [newWidth, newHeight, 3])

        for y in range(newHeight):
            for x in range(newWidth):
                xNearest = (int)(round(x / scaleX))
                yNearest = (int)(round(y / scaleY))
                xNearest = width - 1 if xNearest >= width else xNearest
                xNearest = 0 if xNearest < 0 else xNearest
                yNearest = height - 1 if yNearest >= height else yNearest
                yNearest = 0 if yNearest < 0 else yNearest
                data[x, y, :] = data_temp[xNearest, yNearest, :]

    def resizeBilinear(self, scaleX, scaleY):
        global data
        global width
        global height
        newWidth = (int)(round(width * scaleX))
        newHeight = (int)(round(height * scaleY))
        data_temp = np.zeros([width, height, 3])
        data_temp = data.copy()
        data = np.resize(data, [newWidth, newHeight, 3])
        for y in range(newHeight):
            for x in range(newWidth):
                oldX = x / scaleX
                oldY = y / scaleY
                #get 4 coordinates
                x1 = min((int)(np.floor(oldX)), width - 1)
                y1 = min((int)(np.floor(oldY)), height - 1)
                x2 = min((int)(np.ceil(oldX)), width - 1)
                y2 = min((int)(np.ceil(oldY)), height - 1)
                #get colours
                color11 = np.array(data_temp[x1, y1, :])
                color12 = np.array(data_temp[x1, y2, :])
                color21 = np.array(data_temp[x2, y1, :])
                color22 = np.array(data_temp[x2, y2, :])
                #interpolate x
                P1 = (x2 - oldX) * color11 + (oldX - x1) * color21
                P2 = (x2 - oldX) * color12 + (oldX - x1) * color22
                
                if x1 == x2:
                    P1 = color11
                    P2 = color22

                #interpolate y
                P = (y2 - oldY) * P1 + (oldY - y1) * P2

                if y1 == y2:
                    P = P1

                P = np.round(P)
                data[x, y, :] = P

    def erosion(self, se):
        global data
        self.convertToGray()
        data_zeropaded = np.zeros([width + se.width * 2, height + se.height * 2, 3])
        data_zeropaded[se.width - 1:width + se.width - 1, se.height - 1:height + se.height - 1, :] = data
        for y in range(se.height - 1, se.height + height - 1):
            for x in range(se.width - 1, se.width + width - 1):
                subData = data_zeropaded[x - int(se.origin.real):x - int(se.origin.real) + se.width, y - int(se.origin.imag):y - int(se.origin.imag) + se.height, 0:1]

                subData = subData.reshape(3, -1)
                
                for point in se.ignoreElements:
                    subData[int(point.real), int(point.imag)] = se.elements[int(point.real),int(point.imag)]

                min = np.amin(se.elements[se.elements > 0])
                if (0 <= x - int(se.origin.real) - 1 < width and 0 <= y - int(se.origin.imag) - 1 < height):

                    if (np.array_equal(subData, se.elements)):
                        data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 0] = min
                        data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 1] = min
                        data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 2] = min
                    else:
                        data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 0] = 0
                        data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 1] = 0
                        data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 2] = 0

    def dilation(self, se):
        global data
        
        self.convertToGray()
        
        data_zeropaded = np.zeros([width + se.width * 2, height + se.height * 2, 3])
        data_zeropaded[se.width - 1:width + se.width - 1, se.height - 1:height + se.height - 1, :] = data
        
        for y in range(se.height - 1, se.height + height - 1):
            for x in range(se.width - 1, se.width + width - 1):
                subData = data_zeropaded[x - int(se.origin.real):x - int(se.origin.real) + se.width, y - int(se.origin.imag):y - int(se.origin.imag) + se.height, 0:1]

                subData = subData.reshape(3, -1)

                for point in se.ignoreElements:
                    subData[int(point.real), int(point.imag)] = se.elements[int(point.real), int(point.imag)]
                
                max = np.amax(se.elements[se.elements > 0])
                subData = np.subtract(subData, np.flip(se.elements))

                if (0 <= x - int(se.origin.real) - 1 < width and 0 <= y - int(se.origin.imag) - 1 < height):

                    if (0 in subData):
                        data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 0] = max
                        data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 1] = max
                        data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 2] = max
                    else:
                        data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 0] = 0
                        data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 1] = 0
                        data[x - int(se.origin.real) - 1, y - int(se.origin.imag) - 1, 2] = 0
    
    # def boundary_extraction(self, se):
    #     eroded_image = self.erosion(se)
    #     dilated_image = self.dilation(se)
        
    #     # Compute boundary by subtracting eroded image from dilated image
    #     boundary = np.clip(dilated_image - eroded_image, 0, 255)
        
    #     # Update self.data with boundary
    #     self.data = boundary.astype(np.uint8)

    def boundary_extraction(self, se):
        global data
        original_data = np.copy(data)

        self.erosion(se)

        for y in range(height):
            for x in range(width):
                for channel in range(3):  # For each RGB channel
                    data[x, y, channel] = original_data[x, y, channel] - data[x, y, channel]

    def thresholding(self, threshold):
        global data
        self.convertToGray()
        for y in range(height):
            for x in range(width):
                gray = data[x, y, 0]
                
                gray = 0 if gray < threshold else 255

                data[x, y, 0] = gray
                data[x, y, 1] = gray
                data[x, y, 2] = gray

    def otsuThreshold(self):
        global data

        self.convertToGray()

        histogram = np.zeros(256)

        for y in range(height):
            for x in range(width):
                histogram[data[x, y, 0]] += 1

        histogramNorm = np.zeros(len(histogram))
        pixelNum = width * height

        for i in range(len(histogramNorm)):
            histogramNorm[i] = histogram[i] / pixelNum

        histogramCS = np.zeros(len(histogram))
        histogramMean = np.zeros(len(histogram))

        for i in range(len(histogramNorm)):
            if (i == 0):
                histogramCS[i] = histogramNorm[i]
                histogramMean[i] = 0
            else:
                histogramCS[i] = histogramCS[i - 1] + histogramNorm[i]
                histogramMean[i] = histogramMean[i - 1] + histogramNorm[i] * i
        
        globalMean = histogramMean[len(histogramMean) - 1]
        max = sys.float_info.min
        maxVariance = sys.float_info.min
        countMax = 0
        
        for i in range(len(histogramCS)):
            if (histogramCS[i] < 1 and histogramCS[i] > 0):
                variance = ((globalMean * histogramCS[i] - histogramMean[i]) ** 2) / (histogramCS[i] * (1 - histogramCS[i]))

            if (variance > maxVariance):
                maxVariance = variance
                max = i
                countMax = 1
            elif (variance == maxVariance):
                countMax = countMax + 1
                max = ((max * (countMax - 1)) + i) / countMax

        self.thresholding(round(max))

    def linearSpatialFilter(self, kernel, size):
        global data

        if (size % 2 ==0):
            print("Size Invalid: must be odd number!")
            return
        
        data_zeropaded = np.zeros([width + int(size/2) * 2, height + int(size/2) * 2, 3])
        data_zeropaded[int(size/2):width + int(size/2), int(size/2):height + int(size/2), :] = data
        for y in range(int(size/2), int(size/2) + height):
            for x in range(int(size/2), int(size/2) + width):
                subData = data_zeropaded[x - int(size/2):x + int(size/2) + 1, y - int(size/2):y + int(size/2) + 1, :]

                sumRed = np.sum(np.multiply(subData[:,:,0:1].flatten(), kernel))
                sumGreen = np.sum(np.multiply(subData[:,:,1:2].flatten(), kernel))
                sumBlue = np.sum(np.multiply(subData[:,:,2:3].flatten(), kernel))
                
                sumRed = 255 if sumRed > 255 else sumRed
                sumRed = 0 if sumRed < 0 else sumRed
                
                sumGreen = 255 if sumGreen > 255 else sumGreen
                sumGreen = 0 if sumGreen < 0 else sumGreen
                
                sumBlue = 255 if sumBlue > 255 else sumBlue
                sumBlue = 0 if sumBlue < 0 else sumBlue
                
                data[x - int(size/2), y - int(size/2), 0] = sumRed
                data[x - int(size/2), y - int(size/2), 1] = sumGreen
                data[x - int(size/2), y - int(size/2), 2] = sumBlue   

    def cannyEdgeDetector(self, lower, upper):
        global data
        #Step 1 - Apply 5 x 5 Gaussian filter
        gaussian = [2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0,
        4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0, 9.0 / 159.0, 4.0 / 159.0,
        5.0 / 159.0, 12.0 / 159.0, 15.0 / 159.0, 12.0 / 159.0, 5.0 / 159.0,
        4.0 / 159.0, 9.0 / 159.0, 12.0 / 159.0, 9.0 / 159.0, 4.0 / 159.0,
        2.0 / 159.0, 4.0 / 159.0, 5.0 / 159.0, 4.0 / 159.0, 2.0 / 159.0]
        
        self.linearSpatialFilter(gaussian, 5)
        self.convertToGray()

        #Step 2 - Find intensity gradient
        sobelX = [ 1, 0, -1,
                    2, 0, -2,
                    1, 0, -1]
        sobelY = [ 1, 2, 1,
                    0, 0, 0,
                    -1, -2, -1]

        magnitude = np.zeros([width, height])
        direction = np.zeros([width, height])
        
        data_zeropaded = np.zeros([width + 2, height + 2, 3])
        data_zeropaded[1:width + 1, 1:height + 1, :] = data
        
        for y in range(1, height + 1):
            for x in range(1, width + 1):
                gx = 0
                gy = 0
                
                subData = data_zeropaded[x - 1:x + 2, y - 1:y + 2, :]
                
                gx = np.sum(np.multiply(subData[:,:,0:1].flatten(), sobelX))
                gy = np.sum(np.multiply(subData[:,:,0:1].flatten(), sobelY))
                
                magnitude[x - 1, y - 1] = math.sqrt(gx * gx + gy * gy)
                direction[x - 1, y - 1] = math.atan2(gy, gx) * 180 / math.pi

        #Step 3 - Nonmaxima Suppression
        gn = np.zeros([width, height])

        for y in range(1, height + 1):
            for x in range(1, width + 1):
                targetX = 0
                targetY = 0

                #find closest direction
                if (direction[x - 1, y - 1] <= -157.5):
                    targetX = 1
                    targetY = 0
                elif (direction[x - 1, y - 1] <= -112.5):
                    targetX = 1
                    targetY = -1
                elif (direction[x - 1, y - 1] <= -67.5):
                    targetX = 0
                    targetY = 1
                elif (direction[x - 1, y - 1] <= -22.5):
                    targetX = 1
                    targetY = 1
                elif (direction[x - 1, y - 1] <= 22.5):
                    targetX = 1
                    targetY = 0
                elif (direction[x - 1, y - 1] <= 67.5):
                    targetX = 1
                    targetY = -1
                elif (direction[x - 1, y - 1] <= 112.5):
                    targetX = 0
                    targetY = 1
                elif (direction[x - 1, y - 1] <= 157.5):
                    targetX = 1
                    targetY = 1
                else:
                    targetX = 1
                    targetY = 0

                if (y + targetY >= 0 and y + targetY < height and x + targetX >= 0 and x + targetX < width and magnitude[x - 1, y - 1] < magnitude[x + targetY - 1, y + targetX - 1]):
                    gn[x - 1, y - 1] = 0
                elif (y - targetY >= 0 and y - targetY < height and x - targetX >= 0 and x - targetX < width and magnitude[x - 1, y - 1] < magnitude[x - targetY - 1, y - targetX - 1]):
                    gn[x - 1, y - 1] = 0
                else:
                    gn[x - 1, y - 1] = magnitude[x - 1, y - 1]
                
                #set back first
                gn[x - 1, y - 1] = 255 if gn[x - 1, y - 1] > 255 else gn[x - 1, y - 1]
                gn[x - 1, y - 1] = 0 if gn[x - 1, y - 1] < 0 else gn[x - 1, y - 1]
                
                data[x - 1, y - 1, 0] = gn[x - 1, y - 1]
                data[x - 1, y - 1, 1] = gn[x - 1, y - 1]
                data[x - 1, y - 1, 2] = gn[x - 1, y - 1]


        #upper threshold checking with recursive
        for y in range(height):
            for x in range(width):
                if (data[x, y, 0] >= upper):
                    data[x, y, 0] = 255
                    data[x, y, 1] = 255
                    data[x, y, 2] = 255

                    self.hystConnect(x, y, lower)

        #clear unwanted values
        for y in range(height):
            for x in range(width):
                if (data[x, y, 0] != 255):
                    data[x, y, 0] = 0
                    data[x, y, 1] = 0
                    data[x, y, 2] = 0

    def hystConnect(self, x, y, threshold):
        global data

        for i in range(y - 1, y + 2):
            for j in range(x - 1, x + 2):
                if ((j < width) and (i < height) and
                    (j >= 0) and (i >= 0) and
                    (j != x) and (i != y)):

                    value = data[j, i, 0]
                    if (value != 255):
                        if (value >= threshold):
                            data[j, i, 0] = 255
                            data[j, i, 1] = 255
                            data[j, i, 2] = 255

                            self.hystConnect(j, i, threshold)
                        else:
                            data[j, i, 0] = 0
                            data[j, i, 1] = 0
                            data[j, i, 2] = 0

    def houghTransform(self, percent):
        global data

        #The image should be converted to edge map first
        
        #Work out how the hough space is quantized
        numOfTheta = 720
        thetaStep = math.pi / numOfTheta
        
        highestR = int(round(max(width, height) * math.sqrt(2)))
        
        centreX = int(width / 2)
        centreY = int(height / 2)
        
        print("Hough array w: %s height: %s" % (numOfTheta, (2*highestR)))
        
        #Create the hough array and initialize to zero
        houghArray = np.zeros([numOfTheta, 2*highestR])
        
        #Step 1 - find each edge pixel
        #Find edge points and vote in array
        for y in range(3, height - 3):
            for x in range(3, width - 3):
                pointColor = data[x, y, 0]
                if (pointColor != 0):
                    #Edge pixel found
                    for i in range(numOfTheta):
                        #Step 2 - Apply the line equation and update hough array
                        #Work out the r values for each theta step
                        
                        r = int((x - centreX) * math.cos(i * thetaStep) + (y - centreY) * math.sin(i * thetaStep))

                    #Move all values into positive range for display purposes
                        r = r + highestR
                        if (r < 0 or r >= 2 * highestR):
                            continue
                        
                        #Increment hough array
                        houghArray[i, r] = houghArray[i, r] + 1
        #Step 3 - Apply threshold to hough array to find line
        #Find the max hough value for the thresholding operation
        maxHough = np.amax(houghArray)

        #Set the threshold limit
        threshold = percent * maxHough

        #Step 4 - Draw lines
        # Search for local peaks above threshold to draw
        for i in range(numOfTheta):
            for j in range(2 * highestR):
                #only consider points above threshold
                if (houghArray[i, j] >= threshold):
                    # see if local maxima
                    draw = True
                    peak = houghArray[i, j]
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            #not seeing itself
                            if (k == 0 and l == 0):
                                continue
                            testTheta = i + k
                            testOffset = j + l
                            
                            if (testOffset < 0 or testOffset >= 2*highestR):
                                continue
                            if (testTheta < 0):
                                testTheta = testTheta + numOfTheta
                            if (testTheta >= numOfTheta):
                                testTheta = testTheta - numOfTheta
                            if (houghArray[testTheta][testOffset] > peak):
                                #found bigger point
                                draw = False
                                break

                    #point found is not local maxima
                    if (not(draw)):
                        continue
                    #if local maxima, draw red back
                    tsin = math.sin(i*thetaStep)
                    tcos = math.cos(i*thetaStep)
                    
                    if (i <= numOfTheta / 4 or i >= (3 * numOfTheta) / 4):
                        for y in range(height):
                            #vertical line
                            x = int((((j - highestR) - ((y - centreY) * tsin)) / tcos) + centreX)

                            if(x < width and x >= 0):
                                data[x, y, 0] = 255
                                data[x, y, 1] = 0
                                data[x, y, 2] = 0

                    else:
                        for x in range(width):
                            #horizontal line
                            y = int((((j - highestR) - ((x - centreX) * tcos)) / tsin) + centreY)

                            if(y < height and y >= 0):
                                data[x, y, 0] = 255
                                data[x, y, 1] = 0
                                data[x, y, 2] = 0

    def ADIAbsolute(self, sequences, threshold, step):
        global data
        
        data_temp = np.zeros([width, height, 3])
        data_temp = np.copy(data)
        
        data[data > 0] = 0

        for n in range(len(sequences)):
            #read file
            otherImage = Image.open(sequences[n])
            otherData = np.array(otherImage)
            
            for y in range(height):
                for x in range(width):
                    dr = int(data_temp[x, y, 0]) - int(otherData[x, y, 0])
                    dg = int(data_temp[x, y, 1]) - int(otherData[x, y, 1])
                    db = int(data_temp[x, y, 2]) - int(otherData[x, y, 2])
                    
                    dGray = int(round((0.2126*dr) + int(0.7152*dg) + int(0.0722*db)))
                    
                    if (abs(dGray) > threshold):
                        newColor = data[x, y, 0] + step
                        
                        newColor = 255 if newColor > 255 else newColor
                        newColor = 0 if newColor < 0 else newColor
                        
                        data[x, y, 0] = newColor
                        data[x, y, 1] = newColor
                        data[x, y, 2] = newColor

    def detectHarrisFeatures(self, strongest):
        global data
        # Convert to grayscale
        self.convertToGray()
        
        # Compute gradients Ix and Iy, drop the border
        Ix = np.zeros((height, width), dtype=np.float32)
        Iy = np.zeros((height, width), dtype=np.float32)
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                Ix[y, x] = (float(data[x + 1, y, 0]) - float(data[x - 1, y, 0])) / 2.0
                Iy[y, x] = (float(data[x, y + 1, 0]) - float(data[x, y - 1, 0])) / 2.0
        
        # Initialize matrices to store products of gradients
        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        Ixy = Ix * Iy

        # Apply 3x3 Gaussian smoothing
        gaussian = np.array([
            [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0],
            [2.0 / 16.0, 4.0 / 16.0, 2.0 / 16.0],
            [1.0 / 16.0, 2.0 / 16.0, 1.0 / 16.0]])
        
        Sx2 = np.zeros((height, width), dtype=np.float32)
        Sy2 = np.zeros((height, width), dtype=np.float32)
        Sxy = np.zeros((height, width), dtype=np.float32)
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                Sx2[y, x] = np.sum(Ix2[y - 1:y + 2, x - 1:x + 2] * gaussian)
                Sy2[y, x] = np.sum(Iy2[y - 1:y + 2, x - 1:x + 2] * gaussian)
                Sxy[y, x] = np.sum(Ixy[y - 1:y + 2, x - 1:x + 2] * gaussian)

        # Compute the corner response function R
        corners = np.zeros((height, width))
        for y in range(height):
            for x in range(width):
                det = Sx2[y, x] * Sy2[y, x] - Sxy[y, x] * Sxy[y, x]
                trace = Sx2[y, x] + Sy2[y, x]
                corners[y, x] = det - 0.04 * (trace ** 2)

        cornerPoints = []
        cornerValues = []

        # Maxima Suppression
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if corners[y, x] < 0:
                    continue

                peak = corners[y, x]
                isMaxima = True

                # Check 3x3 neighborhood
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        if k == 0 and l == 0:
                            continue
                        if corners[y + k, x + l] > peak:
                            isMaxima = False
                            break
                    if not isMaxima:
                        break
                
                if isMaxima:
                    insertPos = 0
                    while insertPos < len(cornerValues) and cornerValues[insertPos] > peak:
                        insertPos += 1
                    
                    cornerPoints.insert(insertPos, (x, y))
                    cornerValues.insert(insertPos, peak)
                    
                    if len(cornerPoints) > strongest:
                        cornerPoints.pop()
                        cornerValues.pop()
            
        # Draw red X on the image at the corner points
        for p in cornerPoints:
            data[p[0], p[1], 0] = 255
            data[p[0], p[1], 1] = 0
            data[p[0], p[1], 2] = 0

            data[p[0] + 1, p[1] + 1, 0] = 255
            data[p[0] + 1, p[1] + 1, 1] = 0
            data[p[0] + 1, p[1] + 1, 2] = 0

            data[p[0] + 1, p[1] - 1, 0] = 255
            data[p[0] + 1, p[1] - 1, 1] = 0
            data[p[0] + 1, p[1] - 1, 2] = 0

            data[p[0] - 1, p[1] + 1, 0] = 255
            data[p[0] - 1, p[1] + 1, 1] = 0
            data[p[0] - 1, p[1] + 1, 2] = 0

            data[p[0] - 1, p[1] - 1, 0] = 255
            data[p[0] - 1, p[1] - 1, 1] = 0
            data[p[0] - 1, p[1] - 1, 2] = 0
        
        return cornerPoints
    
    def calculateHomography(self, srcPoints, dstPoints):
        A = np.zeros((8, 8))
        b = np.zeros(8)

        for i in range(4):
            xSrc, ySrc = srcPoints[i]
            xDst, yDst = dstPoints[i]

            A[2 * i] = [xSrc, ySrc, 1, 0, 0, 0, -xSrc * xDst, -ySrc * xDst]
            A[2 * i + 1] = [0, 0, 0, xSrc, ySrc, 1, -xSrc * yDst, -ySrc * yDst]
            b[2 * i] = xDst
            b[2 * i + 1] = yDst

        # Solve using Gaussian elimination
        # This function will solve the system A * x = b
        # You can use Gaussian elimination, LU decomposition, or any other method
        return self.gaussianElimination(A, b)
    
    def gaussianElimination(self, A, b):
        n = len(b)
        for i in range(n):
            # Pivoting
            maxIndex = np.argmax(np.abs(A[i:, i])) + i
            A[[i, maxIndex]] = A[[maxIndex, i]]
            b[i], b[maxIndex] = b[maxIndex], b[i]
            
            # Normalize the row
            for k in range(i + 1, n):
                factor = A[k, i] / A[i, i]
                b[k] -= factor * b[i]
                A[k, i:] -= factor * A[i, i:]
            
        # Back substitution
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
            
        # The last element of the homography matrix (h33) is 1
        homography = np.zeros(9)
        homography[:8] = x
        homography[8] = 1

        return homography
    
    def invertHomography(self, H):
        # Calculate the determinant of the 3x3 matrix
        det = (H[0] * (H[4] * H[8] - H[5] * H[7])
            - H[1] * (H[3] * H[8] - H[5] * H[6])
            + H[2] * (H[3] * H[7] - H[4] * H[6]))
        
        if det == 0:
            raise ValueError("Matrix is not invertible")
    
        invDet = 1.0 / det

        # Calculate the inverse using the cofactor matrix
        invH = np.zeros(9)
        invH[0] = invDet * (H[4] * H[8] - H[5] * H[7])
        invH[1] = invDet * (H[2] * H[7] - H[1] * H[8])
        invH[2] = invDet * (H[1] * H[5] - H[2] * H[4])

        invH[3] = invDet * (H[5] * H[6] - H[3] * H[8])
        invH[4] = invDet * (H[0] * H[8] - H[2] * H[6])
        invH[5] = invDet * (H[2] * H[3] - H[0] * H[5])

        invH[6] = invDet * (H[3] * H[7] - H[4] * H[6])
        invH[7] = invDet * (H[1] * H[6] - H[0] * H[7])
        invH[8] = invDet * (H[0] * H[4] - H[1] * H[3])

        return invH
    
    def applyHomographyToPoint(self, H, x, y):
        # Homogeneous coordinates calculation after transformation
        xh = H[0] * x + H[1] * y + H[2]
        yh = H[3] * x + H[4] * y + H[5]
        w = H[6] * x + H[7] * y + H[8]

        # Normalize by w to get the Cartesian coordinates in the destination image
        xPrime = xh / w
        yPrime = yh / w

        return np.array([xPrime, yPrime])
    
    def applyHomography(self, H):
        global data

        data_temp = np.zeros([height, width, 3])
        data_temp = np.copy(data)

        invH = self.invertHomography(H)

        for y in range(height):
            for x in range(width):
                # Apply the inverse of the homography to find the corresponding source pixel
                sourcePoint = self.applyHomographyToPoint(invH, x, y)
                
                srcX = int(round(sourcePoint[0]))
                srcY = int(round(sourcePoint[1]))
                
                # Check if the calculated source coordinates are within the source image bounds

                if 0 <= srcX < width and 0 <= srcY < height:
                    # Copy the pixel from the source image to the destination image
                    data_temp[y, x] = data[srcY, srcX]
                else:
                    # If out of bounds, set the destination pixel to a default color
                    data_temp[y, x] = [0, 0, 0]

        # Copy the processed image back to the original image
        data = np.copy(data_temp)