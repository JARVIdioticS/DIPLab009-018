from ImageManager import ImageManager
from StructuringElement import StructuringElement
import numpy as np
im = ImageManager()

im.read("images/mandril.bmp")
# im.read("images/gamemaster_noise_2024.bmp")

# Quest 009
# im.addSaltAndPepperNoise(0.1) # 0.1 = 10%

# im.write("images/mandrilSaltandPepper10.bmp")

# Quest 010
# im.read("images/mandrilSaltandPepper10.bmp")

# im.contraharmonicFilter( 3, -1.5)
# im.write("images/mandrilCHM3_-1.5.bmp")

# im.restoreToOriginal()


# im.contraharmonicFilter( 3, 1.5)
# im.write("images/mandrilCHM3_1.5.bmp")

# Quest 011
# im.resizeBilinear(3.5, 3.5)
# im.write("images/mandril3.5x.bmp")
# im.restoreToOriginal()
# im.resizeBilinear(0.35, 0.35)
# im.write("images/mandril0.35x.bmp")

# Quest 012
# im.read("images/mandrilB.bmp")
# se = StructuringElement(3, 3, complex(1, 1))
# se.elements[True] = 255
# # im.boundary_extraction(se)
# im.write("images/mandrilboundary.bmp")

# Quest 013
# im.otsuThreshold()
# im.write("images/mandrilOtsu.bmp")

# im.cannyEdgeDetector(100, 180)
# im.write("images/mandrilCannyL100_U180.bmp")

# Quest 014
# im.cannyEdgeDetector(48, 128)
# im.houghTransform(0.8)
# im.write("images/mandrilCannyHough0.8.bmp")

# Quest 015
# im.read("images/4/motion01.512.bmp")
# seq = ['images/4/motion01.512.bmp','images/4/motion02.512.bmp','images/4/motion03.512.bmp',
#        'images/4/motion04.512.bmp','images/4/motion05.512.bmp','images/4/motion06.512.bmp',
#        'images/4/motion07.512.bmp','images/4/motion08.512.bmp','images/4/motion09.512.bmp',
#        'images/4/motion10.512.bmp']
# im.ADIAbsolute(seq, 25, 50)
# im.write("images/ADIseq4.bmp")

# Quest 016
# im.detectHarrisFeatures(1000)
# im.write("images/mandrilHarris1000.bmp")
# im.restoreToOriginal()
# im.detectHarrisFeatures(2000)
# im.write("images/mandrilHarris2000.bmp")

# Quest 017
# im.read("images/qrcode.bmp")
# # Define source and destination points
# srcPoints = np.array([
# [256, 133], # top-left
# [419, 146], # top-right
# [403, 348], # bottom-right
# [244, 320] # bottom-left
# ])

# dstPoints = np.array([
# [0, 0], # top-left
# [512, 0], # top-right
# [512, 512], # bottom-right
# [0, 512] # bottom-left
# ])


