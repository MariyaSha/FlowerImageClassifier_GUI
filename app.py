#app.py imports (GUI related)
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QGridLayout
from PyQt5.QtGui import QPixmap

#functions.py module imports
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import os
import pandas as pd
import numpy as np
from PIL import Image

#functions.py function imports
from functions import load_checkpoint, process_image, get_prediction, predict

#button "click" event function
def display_prediction():
    '''this function occurs each time the "Browse" button is pressed'''
    btn.setText("Loading...") #notify the user that the app is loading

    #collect user input
    get_file = QFileDialog.getOpenFileName(btn, "Open File", "/home", "Images (*.png *.xpm *.jpg)")
    user_image = get_file[0]

    #get and display prediction
    flower_name = get_prediction(user_image)
    pred.setText("this is a "+ flower_name)

    #display user image
    imgPixmap = QPixmap(user_image)
    if imgPixmap.height() > imgPixmap.width():
        #if image is vertical
        imgPixmap = imgPixmap.scaledToWidth(150)
    else:
        #if image is horizontal
        imgPixmap = imgPixmap.scaledToHeight(150)
    img.setPixmap(imgPixmap)
    img.setAlignment(Qt.AlignCenter)
    img.setStyleSheet(
        "padding: 10px; margin: 10px;"
        )
    btn.setText("Browse") #set button back to 'browse'

app = QApplication(sys.argv)

#window object settings
window = QWidget()
window.setWindowTitle("What the Flower?")
window.setFixedWidth(600)

#page grid
grid = QGridLayout()

#logo settings
pixmap = QPixmap('assets/logo.png')
logo = QLabel()
logo.setPixmap(pixmap)
logo.setAlignment(Qt.AlignCenter)
logo.setStyleSheet(
    "padding :10px; margin-top:30px;"
)

#instructions settings
inst = QLabel("Please select a flower image from your\n computer to see its name")
inst.setAlignment(Qt.AlignCenter)
inst.setStyleSheet(
    "font-family: 'Shanti'; font-size: 21px;"
    )

#browse button settings
btn = QPushButton('Browse')
btn.setStyleSheet(
    "*{padding:10px; margin-top:30px; margin-right:50px; margin-left:50px; background: '#73C059'; color: 'white'; font-family: 'Pacifico'; font-size: 25px; border-radius: 10px;} *:hover{background:'#55933B';}"
    )
#connecting the button to a function via click event
btn.clicked.connect(display_prediction)

#prediction settings
pred = QLabel('')
pred.setAlignment(Qt.AlignCenter)
pred.setStyleSheet("font-family: 'Shanti'; font-size: 21px; padding:10px;")

#image settings (element is hidden until the user selects an image)
img = QLabel('')
img.setStyleSheet(
    "margin: 125px;" #placeholder for the image
    )

#placing windgets on the layout grid
grid.addWidget(logo, 0, 0)
grid.addWidget(inst, 1, 0)
grid.addWidget(btn, 2, 0)
grid.addWidget(pred, 3, 0)
grid.addWidget(img, 4, 0)

#set grid and show window
window.setLayout(grid)
window.show()

sys.exit(app.exec_())
