# -*- coding: utf-8 -*-
"""
Created on Wed May 16 13:20:32 2018

@author: ovv
"""
# import shutil
from tkinter import *
from PIL import Image, ImageTk, ExifTags, ImageFile, ImageGrab, ImageDraw
import pandas as pd
import numpy as np
#from tkintertable import TableCanvas
import math

# THESE MUST GO WHEN CONVERTING TO WINDOWS EXE
#import matplotlib.pyplot as plt
#from matplotlib.figure import Figure
#import matplotlib as mpl
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg#, NavigationToolbar2TkAgg
###### UNTIL HERE


#import networkx as nx
# import pickle
from tkinter import ttk, filedialog, messagebox
import tkinter.simpledialog as simpledialog
import os
import io
import string
import random
# import glob
#import torch
import copy
import webbrowser
# from PIL.ExifTags import TAGS, GPSTAGS
from itertools import chain
# import torch.utils.data as data
ImageFile.LOAD_TRUNCATED_IMAGES = True
import umap
from collections import defaultdict
import functools
import h5py
import h5py.defs
import h5py.utils
import h5py.h5ac
import h5py._proxy
import warnings
#import cv2
import time
#import timm

# import rasterfairy #from 2d point cloud to square grid. Need to change import rasterfairy.prime to import rasterfairy.prime as prime in the rasterfairy.py file. You also need to change np.float to float.
import bottleneck as bn
import exifread
import hypernetx as hnx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import PolyCollection, PathCollection
from matplotlib import colormaps
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)


from sklearn.neighbors import NearestNeighbors
from threading import Timer
from sklearn.metrics.pairwise import cosine_similarity
# from VirtualMatrix import VirtualMatrix, create_frozen_header
from matrix_view import LazyImagePanel, SyncScrollExample, HyperedgeConfusionMatrix, ConfusionMatrixWithImages
from tkintertable import TableCanvas, TableModel
from collections import OrderedDict
from typing import Tuple, List, Optional, Union 
from utils.file_utils import get_image_files
from utils.feature_extraction import FeatureExtractor
# from synced_table_canvas import SyncedTableCanvas
#import clip ## only if model used is clip... 

batch_size  = 8
num_workers = 4
num_c = 0
tuu=-1
#image_= []
def numpy_unique_ordered(seq):
            array_unique = np.unique(seq, return_index=True)
            dstack = np.dstack(array_unique)
            dstack.dtype = np.dtype([('v', dstack.dtype), ('i', dstack.dtype)])
            dstack.sort(order='i', axis=1)
            return dstack.flatten()['v'].tolist()

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
    #id generator for exporting images with the same name.

#Main app window
class Application(Frame,object):
    # Define settings upon initialization. Here you can specify
    def __init__(self, master):
        self.themainframe = super(Application,self).__init__(master)
        #some initial parameters
        self.refresh = 0            #keeps track of whether the graph has been updated recently or not
        self.im_list = []           #list with images
        #self.df = pd.DataFrame()    #clustering results
        self.df = np.array([])    #clustering results
        self.lijstje = []           #list with buckets
        self.theBuckets = []        #the buckets containing the images
        self.bucketDisp = 0         # is 1 when a bucket is displayed, 0 otherwise.
        self.allimages = []         #if images are Preloaded, this contains the images.
        self.num_clus = []          #the number of the current cluster
        #self.cm = []                #the array with correlations between images
        self.features: Optional[np.ndarray] = None           #the features of all the images extracted with the neural network
        self.catList = []           #similar to lijstje
        self.loaded_imgs = []       #contains all the actual image data, if preloaded
        self.imagex = []            #contains the images currently displayed
        self.preloaded = 0          #keeps track of whether images are preloaded or not
        self.meta_load = 0          #keeps track of whether the metatab has been loaded or not
        self.rectangles = []        #contains the squares to display currently selected images
        self.greentangles = []      #contains the squares to display current images that are already in a bucket
        self.selected_images = []   #list of selected images
        #self.grid()
        self.master.title("IMEX")   #title of the app
#        self.image_distance = 10    #distance in pixels between images displayed
        self.focusclick = 0         #keeps track of creating the square on a focused image
        self.rectanglewidth = 3     #width of the red selection rectangles
        self.neuralnet = 'resnet152'#determines the neural network used.
        # self.neuralnet = 'clip'#determines the neural network used.
        self.subcyes = 0            # is 1 when subclustering, 0 otherwise
        self.tsneclick = 0          #keeps track of creating the square on tsne graph
        self.umap_features = []            # variable for UMAP/TSNE embedding
        self.wt = 0
        self.selectedcat = []       #needed to make it possible to deselect item in category box
        self.video_cm = []          #in case the user adds videos
        self.zearray_reduced = []   ### array for image map
        self.log = []
        self.represent = []
        self.scrolled_images = 0
        self.totimseen = []
        self.ccluster = []
        self.oview = 0
        self.controlZ = []
        self.czcurrent = 0
        self.ctrlzused = 0
        self.meta_data = pd.DataFrame() # DataFrame to hold metadata
        self.selected_edge = None # Currently selected hyperedge
        self.hyperedge_canvases = [] # to store our hyperedge canvases of the second window
        self.previous_hypergraphs = []
        self.list_of_colors = ['crimson', 'gold', 'royalblue', 'hotpink', 'turqoise', 'purple', 'lime', 'lightyellow' ,'forestgreen','silver']
        self.text_removes = 0
        self.previous_visible_points = []
        self.image_boxes = []
        self.ctrl_is_held = False
        self.lasso = None
        self.checkbox_vars = {}
        self.confusion_matrix = None
        style = ttk.Style()
        style.theme_use('default')
        style.configure('.',background = '#555555',foreground = 'white') #sets background of the app to a dark theme
        style.map('.',background=[('disabled','#555555'),('active','#777777')], relief=[('!pressed','sunken'),('pressed','raised')])
        style.configure('TNotebook.Tab',background='#555555')
        style.map('TNotebook.Tab',background=[('selected','black')])
#       
        self.temp_image_paths: Optional[List[str]] = None
        self.temp_features: Optional[np.ndarray] = None
        self.temp_thumbnail_data: Optional[Union[List[Optional[bytes]], List[Optional[str]]]] = None
        self.temp_thumbnails_are_embedded: bool = True # Default for new extractions
        self.project_file_path: Optional[str] = None
        self.app_thumbnail_data: Optional[Union[List[Optional[bytes]], List[Optional[str]]]] = None
        self.app_thumbnails_are_embedded: bool = True

        self.store_thumbnails_externally_setting: bool = False # DEFAULT: Embed in HDF5, set to true to store separate thumbnails. Maybe necessary for very large datasets?
        
        self.create_widget()

    def create_widget(self):
        
        #########################################
        ######## ALL tab1 WIDGETS ###############
        #########################################        
        self.image_distance = 10
        self.top_imex_frame = Frame(root,bg="#666666")
        self.top_imex_frame.pack(side="top", fill="x")

        self.topleft_imex_frame = Frame(self.top_imex_frame,bg="#666666")
        self.topleft_imex_frame.pack(side="left")
        
        self.topmiddleleft_imex_frame = Frame(self.top_imex_frame,bg="#666666")
        self.topmiddleleft_imex_frame.pack(side="left")

        self.topmiddleright_imex_frame = Frame(self.top_imex_frame,bg="#666666")
        self.topmiddleright_imex_frame.pack(side="left")

        self.topright_imex_frame = Frame(self.top_imex_frame,bg="#666666")
        self.topright_imex_frame.pack(side="left")

        self.bottom_imex_frame = Frame(root,bg="#666666")
        self.bottom_imex_frame.pack(side="top", fill="both", expand=True)

        self.bottom_imex_frame.grid_rowconfigure(0, weight=1)
        self.bottom_imex_frame.grid_columnconfigure(0, weight=2)  # left 2/3
        self.bottom_imex_frame.grid_columnconfigure(1, weight=1)  # right 1/3
        
        # Place the child frames using grid
        self.bottomleft_imex_frame = Frame(self.bottom_imex_frame, bg="#666666")
        self.bottomleft_imex_frame.grid(row=0, column=0, sticky="nsew")

        self.bottomleft_imex_frame.grid_rowconfigure(0, weight=1)
        self.bottomleft_imex_frame.grid_columnconfigure(0, weight=1)

        self.bottomright_imex_frame = Frame(self.bottom_imex_frame, bg="#666666")
        self.bottomright_imex_frame.grid(row=0, column=1, sticky="nsew")

        
        #button explanation text. The user can right-click a button to get this explanation
        def open_folder_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button you can select the image folder containing the images you want to analyze. IMEX will look through the selected folder and all its subfolders for files with the following extentions: .jpg, .png, .bmp, .gif, .tif')
        def features_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button IMEX will extract the features of the images it found in the selected folder using a neural network. The features are a way to represent the content of an image.')
        def cluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='The features will be used generate clusters based on similarity. Images above the threshold will be placed in the same cluster. If clusters contain a lot of false positives, increasing threshold. If there are too many small clusters, try decreasing the threshold.')
        def load_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Load a previously saved session and continue where you stopped last time.')
        def addcategory_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Add a category bucket to structure the image collection with the name specified in the entry field above. You can add images to the buckets using the \'Add cluster to selected bucket(s)\' button, or the \'Add selection to selected bucket(s)\' button.')
        def showcluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Show a specified cluster number')
        def showbucket_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Show the content of the selected bucket(s). This also allows you to delete images from the buckets and in a future update, use the bucket to find more relevant instances.')
        def export_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Export the content of your buckets to a location of your choosing. A folder will be created for each bucket, and a copy of the original images will be placed in the folders.')
        def showimg_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='This button shows the content of the current cluster. You can set how many images you see in the entry box above this button.')
        def locateonmap_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Shows where an image is located on the image map.')
        def addcluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button you can add your current cluster to one or more category buckets that you have selected from the list in the middle of the screen. This adds the entire cluster to the bucket at once. If you want more control, you can make a selection, and use the \'Add selection to selected bucket(s)\' button.')
        def addselection_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button you can add your selected images to one or more category buckets that you have selected from the list in the middle of the screen.')
        def delete_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button you can delete selected images from a bucket. To do this you first need to use the \'Show bucket\' button, and it only works if you are viewing a single bucket.')
        def changename_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Change the name of the selected bucket to what you put in the entry field below this button')
        def numim_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Here you can set how many images you want to be shown at the same time. The default is 20. Set to zero if you want to view all images in a cluster or bucket. Note that it can take a moment to load all images when the number is high.')
        def vector_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Press this button to calculate a representative image for each cluster. This allows you to use the \'Show overview\' button, which shows a single representative image for each cluster.')
        def imagemap_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Calculates a map with all images, with similar images closer together. Hold shift and draw a shape with left mouse button to select images within the shape.')
        def macromap_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='A \'zoomed out\' version of the image map, use mousewheel on an image to zoom in to view the other images surrounding this image. If you zoom in enough you end up on the actual image map.')
        def overview_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Show the overview after calculating the overview. Displays a representative image for each cluster, which allows you to quickly see relevant clusters. After you have selected one or more clusters, press the \'Show selected cluster\' button to view the cluster.')
        def selectcluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Press this button to view the cluster you selected from the overview. You can always return to the overview by pressing the \'Show overview\' button again.')
        def focus_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='If only a part of the image is of particular interest, you can click this button to enlarge the image. Next, you can draw a square around the part, and press the \'Query selection\' button.')
        def query_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='All images in your collection will be ranked on similarity to the part you selected after pressing the \'Focus image\' button.')
        def rankexternal_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Select an image on your computer. All images in your collection will be ranked on similarity to the specified image. This image does not have to be part of the image collection.')
        def rankbucket_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='All images in your collection will be ranked on similarity to the images in the selected bucket. Makes use of an SVM.')
        def rank_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='All images in your collection will be ranked on similarity to the selected image.')
        def save_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='This button will save your current session. The following things are saved: the list of images from the folder you selected, the buckets you created and the images you added to the buckets, and the cluster you are currently at.')
        def preload_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='This button will load thumbnails of all the images in the folder you selected. This will greatly increase the speed of displaying images, and will result in a more pleasurable experience. However, preloading all images may take some time initially.')
        def nonrelevant_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='If you have added images to the \'NonRelevant\' bucket, you can check this box to make sure the images are not displayed anymore, for example when viewing clusters, or when ranking images.')
        def inbucket_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Check this box to hide all images that you already placed in a bucket. This is useful for example if you are are ranking images, and only want to find new instances of images not yet in a bucket.')
        def filter_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='By selecting one or more buckets from the left list, only images present in all of the selected buckets will be shown. Additionally, by selecting a bucket on the right, images in those buckets will not be shown. The lists function as AND (left) and NOT (right).')
        def sankey_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='With this button a sankey graph can be created. For each bucket, it will show which fraction of the images is also present in other buckets.')
        def sorting_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Sort all clusters from most to least relevant based on the images in the currently viewed bucket')
        def restore_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Undo sorting of clusters, restoring the original order')
        def selectvideo_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='WIP, rudimentary version currently implemented. Select a folder containing videos to add to the image collection. Use \'Extract video\' after. ')
        def exctractvideo_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='WIP, rudimentary version currently implemented. Every key frame for each video will be extracted, then within the video these will be clustered. A representative image of each cluster will be added to the clusters in the image collection.')
        def sourcecluster_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Go to the source cluster of the currently selected image.')
        def prev_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Go to previously shown images.')
        def next_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Go forward in shown images.')
        def newimgs_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Add new images to existing image collection. Currently only works by adding a premade txt files with image locations.')
        def get_metadata_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Retrieve metadata (EXIF) from all images.')
        def display_metadata__text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Open table with selected metadata in new window.')
        def add_metadata_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Select metadata types to add to the metadata table for viewing.')
        def display_metaimages_text(event):
            self.communication_label.configure(text='')
            self.communication_label.configure(text='Sort the metadata in the table (e.g., by DateTimeOriginal). Select a row in the table to display images from that point onward.')
        

        ###################################
        ###      topleft_imex_frame     ###
        ###################################


        self.extract_button = Button(self.topleft_imex_frame, background='#445544',foreground='white',width='20',relief='solid',bd=1)
        self.extract_button['text'] = "Select Folder & Extract"
        self.extract_button['command'] = self.run_extraction
        self.extract_button.pack()

        #button to load
        self.lb = Button(self.topleft_imex_frame,background='#445544',foreground='white',width='20',relief='solid',bd=1)
        self.lb['text'] ="Load session"
        self.lb['command'] = lambda:[self.load_as(),self.display_hyperedges()]
        self.lb.bind('<Button-3>',load_text)
        self.lb.pack()

        #button to change name of a bucket
        self.changebutton = Button(self.topleft_imex_frame,background='#443344',foreground='white',width='20',relief='solid',bd=1)
        self.changebutton['text'] = "Rename hyperedge"
        self.changebutton['command'] = self.rename_hyperedge
        self.changebutton.bind('<Button-3>',changename_text)
        self.changebutton.pack()

        # self.e2 = Entry(self.topleft_imex_frame,background='#777777',foreground = 'white',exportselection=0,width=23)
        # self.e2.insert(END, 'new name')
        # self.e2.pack()

        self.add_he_button = Button(self.topleft_imex_frame,background='#443344',foreground='white',width='20',relief='solid',bd=1)
        self.add_he_button['text'] = "Add hyperedge"
        self.add_he_button['command'] = self.add_hyperedge
        self.add_he_button.bind('<Button-3>',changename_text)
        self.add_he_button.pack()

        self.remove_he_button = Button(self.topleft_imex_frame,background='#443344',foreground='white',width='20',relief='solid',bd=1)
        self.remove_he_button['text'] = "Delete hyperedge"
        self.remove_he_button['command'] = self.delete_hyperedge
        self.remove_he_button.bind('<Button-3>',changename_text)
        self.remove_he_button.pack()

        #button to show currently selected bucket
        self.b4 = Button(self.topleft_imex_frame,background='#443344',foreground='white',width='20',relief='solid',bd=1)
        self.b4['text'] ="Show hyperedge"
        self.b4.bind('<Button-3>',showbucket_text)        
        self.b4['command'] = self.showEdge
        self.b4.pack()
        
        self.b5 = Button(self.topleft_imex_frame,background='#443344', foreground='white', width=20, relief='solid', bd=1,
                wraplength=150)
        self.b5['text'] = "Remove image from hyperedge"
        self.b5.bind('<Button-3>',showbucket_text)        
        self.b5['command'] = self.remove_selected_image_from_hyperedge
        self.b5.pack()

        self.addimgbutton = Button(self.topleft_imex_frame,background='#443344',foreground='white',width='20',relief='solid',bd=1)
        self.addimgbutton['text'] ="Add image to hyperedge"
        self.addimgbutton.bind('<Button-3>',showbucket_text)        
        self.addimgbutton['command'] = self.add_selected_image_to_hyperedge
        self.addimgbutton.pack()

        ###################################
        ###   topmiddleleft_imex_frame  ###
        ###################################

        self.search_var = StringVar()
        self.search_var.trace_add("write", self.update_the_list)
        
        self.filter_cats_label = Label(self.topmiddleleft_imex_frame,width=20,background='#555555',foreground='white',relief='flat',justify=LEFT, anchor='w')
        self.filter_cats_label['text'] = 'Search hyperedge'
        self.filter_cats_label.pack()
        self.filter_entry = Entry(self.topmiddleleft_imex_frame,background='#777777',foreground = 'white', textvariable=self.search_var, width=30)
        self.filter_entry.pack()

        boxscrollbar = Scrollbar(self.topmiddleleft_imex_frame,width = 10)
        #listbox containing the names of the buckets
        self.categories = Listbox(self.topmiddleleft_imex_frame,width=30,background='#777777',foreground='white',yscrollcommand=boxscrollbar.set,exportselection=0)
        # self.categories['listvariable'] = valores
        self.categories['selectmode'] = 'extended'               
        self.categories.pack()
        #place of the scrollbar
        boxscrollbar.config(command=self.categories.yview)
        boxscrollbar.place(in_=self.categories,relx=1.0, relheight=1)
        self.categories.bind('<Button-1>', self.deselect_list )
        self.categories.bind("<Double-Button-1>", self.double_click_bucket)
        #filter the self.categories listbox

        ###################################
        ###  topmiddleright_imex_frame  ###
        ###################################


        # Button to add selected hyperedges to chosen_edges
        self.add_button = Button(
            self.topmiddleright_imex_frame,
            text="Add >",
            command=self.add_to_chosen_edges
        )
        self.add_button.pack()

        # Button to remove selected hyperedges from chosen_edges
        self.remove_button = Button(
            self.topmiddleright_imex_frame,
            text="< Remove",
            command=self.remove_from_chosen_edges
        )
        self.remove_button.pack()

        ###################################
        ###     topright_imex_frame     ###
        ###################################

        #button to show currently selected bucket
        self.button_overview = Button(self.topright_imex_frame,background='#443344',foreground='white',width='20',relief='solid',bd=1)
        self.button_overview['text'] ="Create hypergraph overview"
        self.button_overview.bind('<Button-3>',showbucket_text)        
        self.button_overview['command'] = self.show_representative_overview
        self.button_overview.pack()

        self.chosen_edges = Listbox(
        self.topright_imex_frame,  # or whichever parent
        width=30,
        background='#777777',
        foreground='white',
        selectmode='extended',
        exportselection=False
    )
        self.chosen_edges.pack()


        # Radio buttons for exclude/include
        self.filter_mode = StringVar(self.topright_imex_frame,value="exclude")  # default

        self.exclude_rb = Radiobutton(
            self.topright_imex_frame,
            text="Exclude",
            variable=self.filter_mode,
            value="exclude",
            command=self.apply_filter
        )
        self.exclude_rb.pack(side='left')

        self.include_rb = Radiobutton(
            self.topright_imex_frame,
            text="Include",
            variable=self.filter_mode,
            value="include",
            command=self.apply_filter
        )
        self.include_rb.pack(side='left')



        ### window for matrix view
        self.matrixWindow = Toplevel(self.master)
        self.matrixWindow.geometry("1400x500")
        self.matrixWindow.title("Matrix view")
        self.matrixWindow.configure(background='#555555')
        
        self.matrix_notebook = ttk.Notebook(self.matrixWindow)
        self.matrix_notebook.grid(row=0, column=0, sticky="nsew")

        # Create two frames to act as tabs
        self.tab1 = Frame(self.matrix_notebook, background='#555555')
        self.tab2 = Frame(self.matrix_notebook, background='#555555')

        # Add the tabs to the notebook
        self.matrix_notebook.add(self.tab1, text="Edge View")
        self.matrix_notebook.add(self.tab2, text="Hypergraph View")

        # Configure grid layout in self.matrixWindow so the notebook expands
        self.matrixWindow.grid_columnconfigure(0, weight=1)
        self.matrixWindow.grid_rowconfigure(0, weight=1)
        
        ### some other stuff ####
        #creates window for statistics and other data
        self.newWindow = Toplevel(self.master)
        # Right frame for the graph and checkbox
        self.left_frame = Frame(self.newWindow, bg="#555555", width=200)
        self.left_frame.pack(side=LEFT, fill=Y)

        self.right_frame = Frame(self.newWindow, bg="#555555")
        self.right_frame.pack(side=RIGHT, fill=BOTH, expand=TRUE)

        self.newWindow.geometry("1400x500")
        self.newWindow.title("Hyperedges")
        self.newWindow.configure(background='#555555')
        #create dataframe for the bucket
        self.theBuckets = {}
        #list of available category buckets

            
        #enter the size of the images displayed. 100 is default
        self.imsize = 100
        self.screen_width = root.winfo_screenwidth() #requests screen width
        self.screen_width = self.screen_width-60
        self.screen_height = root.winfo_screenheight() #requests screen height
        self.screen_height = self.screen_height-400
     

        self.c = Canvas(self.bottomleft_imex_frame, bg="#666666")
        self.c.grid(row=0, column=0, sticky="nsew")

        # Create and position the vertical scrollbar
        yscrollbar = Scrollbar(self.bottomleft_imex_frame, orient="vertical", command=self.c.yview)
        yscrollbar.grid(row=0, column=1, sticky="ns")

        # Create and position the horizontal scrollbar
        xscrollbar = Scrollbar(self.bottomleft_imex_frame, orient="horizontal", command=self.c.xview)
        xscrollbar.grid(row=1, column=0, sticky="ew")

        # Configure the canvas to use the scrollbars
        self.c.configure(yscrollcommand=yscrollbar.set, xscrollcommand=xscrollbar.set)
        self.c.my_tag = 'c'

        def on_canvas_configure(event):
            canvas_width = event.width
            self.num_im_row = math.floor(canvas_width / (self.imsize + self.image_distance))
    
        self.c.bind("<Configure>", on_canvas_configure)

        # self.num_im_row = math.floor(self.screen_width / (self.imsize + self.image_distance)) #the total number of images that fit from left to right
        self.num_im_col = math.floor(self.screen_height / (self.imsize + self.image_distance)) #the total number of images that fit from top to bottom
        #binds scrollwheel when scrolling with mouse on canvas
        self.c.bind('<Enter>', self._bound_to_mousewheel)
        self.c.bind('<Leave>', self._unbound_to_mousewheel)
        self.c.bind('<Button-2>', self.open_image3)
        self.c.bind("<Button-1>", self.click_select)
        self.c.bind("<Shift-Button-1>", self.shift_click_select)
        self.c.bind("<Control-Button-1>", self.ctrl_click_select)
        # self.c.bind("<Button-3>", self.rank_images)
        self.c.bind("<Double-Button-1>", self.double_click_overview)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.origX = self.c.xview()[0]
        self.origY = self.c.yview()[0]

        linkvv = IntVar()
        self.linked = ttk.Checkbutton(variable = linkvv, master=self.right_frame)
        self.linked['text'] ="Link map with grid"
        self.linked.bind('<Button-3>',nonrelevant_text)        
        self.linked.var = linkvv
        self.linked.pack(side=BOTTOM, pady=10)

        
       
    def confirm_remove_images(self):
        confirm = messagebox.askokcancel("Confirmation", "Make sure to backup the save file first. Are you sure you want to remove images? This action cannot be undone.")
        if confirm:
            self.remove_images()
            
        
  

    def deselect_list(self, event):
        if len(self.selectedcat) == 0:
            self.selectedcat = self.categories.curselection()
        if len(self.selectedcat) == 1:
            if self.categories.curselection() == self.selectedcat:
                self.categories.selection_clear(0,END)
                self.selectedcat = []        
        
    def _bound_to_mousewheel(self,event):
        self.c.bind_all("<MouseWheel>", self._on_mousewheel)
            
    def _unbound_to_mousewheel(self, event):
        self.c.unbind_all("<MouseWheel>") 

    def _bound_to_mousewheel_second_window(self,event):
        event.widget.bind_all("<MouseWheel>", self._on_mousewheel_second_window)
            
    def _unbound_to_mousewheel_second_window(self, event):
        event.widget.unbind_all("<MouseWheel>")

    def update_the_list(self,*args):
        search_term = self.search_var.get()

        self.categories.delete(0, END)
        
        for item in self.catList:
                if search_term.lower() in item.lower():
                    self.categories.insert(END, item)
    
    def open_image3(self, event):
        canvas = event.widget
        evex = canvas.canvasx(event.x)
        evey = canvas.canvasy(event.y)
        self.item = canvas.find_overlapping(evex, evey, evex, evey)
        im_tag = int(canvas.gettags(self.item)[0].split('_')[1])
        mijn_plaatje = self.im_list[im_tag]
        os.startfile(mijn_plaatje)  # Opens image with its default associated viewer
#
    
    def click_select(self,event):
        if self.bucketDisp == 5:
            self.draw_focus_square(event)
        else:
            try:
                self.select_image(event)
            except IndexError:
                pass
    
    def draw_focus_square(self, event):
        self.focusclick = self.focusclick + 1
        if self.focusclick == 2:
            self.evex2 = self.c.canvasx(event.x)
            self.evey2 = self.c.canvasy(event.y)
            self.focusclick = 0
            if self.squares is not None:
                self.c.delete(self.squares)
            self.squares = self.c.create_rectangle(self.evex1, self.evey1, self.evex2, self.evey2)
        else:
            self.evex1 = self.c.canvasx(event.x)
            self.evey1 = self.c.canvasy(event.y)
    
    def select_image(self, event):
        canvas = event.widget
        evex = canvas.canvasx(event.x)
        evey = canvas.canvasy(event.y)
        self.selected_images = []
        # Traverse up to find the canvas
        while not isinstance(canvas, Canvas):
            canvas = canvas.master
        
        # Now you can access the canvas and its tags
        self.canvas_tag = getattr(canvas, 'my_tag', None)
        self.item = event.widget.find_overlapping(evex, evey, evex, evey)
        if self.canvas_tag is not None:
            self.c.delete('rect_tag')
            for canv in self.hyperedge_canvases:
                canv.delete('rect_tag')
            bbox = canvas.bbox(self.item)
            
            # Draw a rectangle around the image to highlight it
            rect = canvas.create_rectangle(
                bbox,
                outline='red',
                width=2,
                tags='rect_tag'
            )
            self.rectangles = [rect]
            self.selected_images.append(canvas.gettags(self.item)[0])

    # this allows you to select multiple adjacent images using the shift key + mouse button 1
    def shift_click_select(self,event):
        canvas = event.widget
        evex = canvas.canvasx(event.x)
        evey = canvas.canvasy(event.y)
        
        # Traverse up to find the canvas
        while not isinstance(canvas, Canvas):
            canvas = canvas.master
        # Now you can access the canvas and its tags
        self.canvas_tag2 = getattr(canvas, 'my_tag', None)
        self.item2 = event.widget.find_overlapping(evex, evey, evex, evey)
        
        if self.canvas_tag2 == self.canvas_tag:
            self.selected_images = []
            self.rectangles = []
            maxit = np.max([self.item[0],self.item2[0]]) + 1
            minit = np.min([self.item[0],self.item2[0]])
            for idx in range(minit,maxit):
                bbox = canvas.bbox(idx)
                rect = canvas.create_rectangle(
                    bbox,
                    outline='red',
                    width=2,
                    tags='rect_tag'
                )
                self.rectangles.append(rect)
                self.selected_images.append(canvas.gettags(idx)[0])

    #this function allows you to select multiple non-adjacent images by hold down control + left click
    def ctrl_click_select(self,event):
        canvas = event.widget
        evex = canvas.canvasx(event.x)
        evey = canvas.canvasy(event.y)
        # Traverse up to find the canvas
        while not isinstance(canvas, Canvas):
            canvas = canvas.master
        # Now you can access the canvas and its tags
        self.canvas_tag = getattr(canvas, 'my_tag', None)
        self.item = event.widget.find_overlapping(evex, evey, evex, evey)
        if canvas.gettags(self.item)[0] in self.selected_images:
            item_to_remove = self.selected_images.index(canvas.gettags(self.item)[0])
            canvas.delete(self.rectangles[item_to_remove])
            del self.selected_images[item_to_remove]
            del self.rectangles[item_to_remove]
            
        else:
            self.item = event.widget.find_overlapping(evex, evey, evex, evey)
            
            if self.canvas_tag is not None:
                bbox = canvas.bbox(self.item)
                
                # Draw a rectangle around the image to highlight it
                rect = canvas.create_rectangle(
                    bbox,
                    outline='red',
                    width=2,
                    tags='rect_tag'
                )
                self.rectangles.append(rect)
                self.selected_images.append(canvas.gettags(self.item)[0])

        
    

    def double_click_bucket(self, event):
        self.showEdge()


    def remove_elements(self, elements):
        self.ccluster = np.array([x for x in self.ccluster if x not in elements])

 
    #the function that is called by other functions in order to display images
    def display_images(self, cluster, input_origin=None):
        self.c.unbind("<ButtonRelease-1>")
        self.c.unbind("<B1-Motion>")
        self.c.unbind("<Shift-B1-Motion>")
        self.c.unbind("<Shift-ButtonRelease-1>")
        self.c.unbind("<Double-Button-1>")

        num_im = min(1000, len(cluster))
        self.ccluster = cluster[:num_im]
        self.c.delete("all")

        # Drop references to old PhotoImages so they can be GC'd
        if hasattr(self, 'my_img'):
            self.my_img.clear()
        else:
            self.my_img = []

        if hasattr(self, 'imagex'):
            self.imagex.clear()
        else:
            self.imagex = []

        # Calculate scroll region
        num_rows = math.ceil(len(self.ccluster) / self.num_im_row)


        self.c['scrollregion'] = (0, 0, 0, num_rows * (self.imsize + self.image_distance))
        # self.c.delete("all")

        with h5py.File(self.hdf_path, 'r') as hdf:
            for j, img_id in enumerate(self.ccluster):
                img_array = np.array(hdf.get('thumbnail_images')[img_id], dtype='uint8')
                load = Image.fromarray(img_array)
                render = ImageTk.PhotoImage(load)
                self.my_img.append(render)

                row_ = j // self.num_im_row
                col_ = j % self.num_im_row
                x_pos = col_ * (self.imsize + self.image_distance) + (self.imsize / 2)
                y_pos = row_ * (self.imsize + self.image_distance) + (self.imsize / 2)

                if self.oview == 1:
                    # Create a tag that includes the hyperedge key, e.g. "edge_<key>"
                    hyperedge_tag = f"edge_{self.hyperedge_rep_name[j]}"
                    tags = (f"img_{img_id}", hyperedge_tag)
                else:
                    tags = (f"img_{img_id}",)


                self.imagex.append(self.c.create_image(x_pos, y_pos, image=render, tags=tags))
            
                if self.oview == 1:
                    # Calculate position for text below the image
                    txt_x = col_ * (self.imsize + self.image_distance) + (self.imsize / 2)
                    txt_y = row_ * (self.imsize + self.image_distance) + self.imsize - (self.image_distance / 2)
                    txtov = self.c.create_text(txt_x, txt_y,
                                            anchor='nw',
                                            fill="white",
                                            font="Calibri 8",
                                            text=str(self.hyperedge_size[j]))
                    # Create a background rectangle behind the text and lower it behind the text
                    txtbb = self.c.create_rectangle(self.c.bbox(txtov), fill="#666666", outline="")
                    self.c.tag_lower(txtbb, txtov)
                    self.c.bind("<Double-Button-1>", self.double_click_overview)

    def double_click_overview(self, event):
        print('extraterrestrial')
        if self.oview == 1:
            print('extraterrestrial 2')
            x, y = event.x, event.y
            clicked_item = self.c.find_closest(x, y)            
            if clicked_item:
                tags = self.c.gettags(clicked_item)
                print(tags)
                # Look for a tag that starts with "edge_"
                for tag in tags:
                    if tag.startswith("edge_"):
                        # Extract the hyperedge key from the tag (removes the "edge_" prefix)
                        hyperedge_key = tag.split("edge_", 1)[1]
                        self.c.delete("all")
                        self.oview = 0
                        self.selected_edge = hyperedge_key
                        self.current_edge = self.selected_edge
                        self.display_hyperedges()
                        break
            

    def _on_mousewheel(self, event):
        self.c.yview_scroll((int(-1*event.delta/120)), "units")
        self.scrolled_images = self.c.yview()
        
         
        
    def _on_mousewheel_second_window(self, event):
        # Windows only
        event.widget.yview_scroll(int(-1*(event.delta/120)), "units")
        
        
    #function to update the image size if the user changes it
    def get_imsize(self):
        self.imsize = self.imsize_entry.get()
        try:
            self.imsize = int(self.imsize)
            if int(self.imsize) > 9:
                self.communication_label.configure(text='Image size set to ' + str(self.imsize))
                self.communication_label['background'] = '#FFFFFF'
                self.num_im_row = math.floor(self.screen_width / (self.imsize + self.image_distance))
                return True
            else:
                self.communication_label.configure(text='Image size needs to be 10 or higher')
                self.communication_label['background'] = '#FE9696'
                self.imsize = 10
                return True
            
        except ValueError:
            self.communication_label.configure(text='Please enter an integer!')
            self.communication_label['background'] = '#FE9696'
            return False

    #function to delete selected images from a bucket
    def click_del(self):
        if len(self.BucketSel) == 1:
            self.catList = self.categories.get(0,END)
            if self.bucketDisp == 1:
                for g in range(0,len(self.selected_images)):
                    self.c.delete(self.imagex[self.selected_images[g]])
                    self.current_bucket[self.selected_images[g]] = -1
                self.current_bucket = np.array(self.current_bucket)
                self.current_bucket = self.current_bucket[self.current_bucket > -1]
                self.theBuckets[self.catList[int(self.BucketSel[0])]] = self.current_bucket


    def create_scenes_from_samples(self, sample_indices, image_mapping):
        """
        Creates a dictionary mapping each hyperedge name -> tuple of images
        from the sample_indices that belong to that hyperedge.
        """
        sample_indices = self.all_filtered_images[sample_indices]
        # scenes is a dict of {hyperedge_name : set_of_images}
        scenes = defaultdict(set)

        # This might be a dict, or maybe an OrderedDict, depending on how you store it
        valid_hyperedges = set(self.filtered_hyperedges.keys())
        if "orphaned_images" in valid_hyperedges:
            valid_hyperedges.remove("orphaned_images")
        # Loop through each image in sample_indices
        for image in sample_indices:
            # Check if the image exists in image_mapping
            if image in image_mapping:
                # Loop through all hyperedges the image belongs to
                for hyperedge_name in image_mapping[image]:
                    # Only include this hyperedge if it's in 'filtered_hyperedges'
                    if hyperedge_name in valid_hyperedges:
                        scenes[hyperedge_name].add(image)

        # Convert sets to sorted tuples
        scenes = {
            h_name: tuple(sorted(imgs))
            for h_name, imgs in scenes.items()
        }

        return scenes




    def create_overview(self):
        if hasattr(self, 'canvasG') and self.canvasG is not None:
            self.canvasG.get_tk_widget().destroy()
            self.canvasG = None  # not strictly required, but helps avoid stale references
        # Initialize UMAP if not done
        if len(self.umap_features) < 1:
            modelu = umap.UMAP()
            self.umap_features = modelu.fit_transform(self.features)
            
            # Normalize UMAP features to fit [0, 1] range
            self.umap_features[:, 0] = self.umap_features[:, 0] + abs(np.min(self.umap_features[:, 0]))
            self.umap_features[:, 1] = self.umap_features[:, 1] + abs(np.min(self.umap_features[:, 1]))
            self.umap_features[:, 0] = self.umap_features[:, 0] / np.max(self.umap_features[:, 0])
            self.umap_features[:, 1] = self.umap_features[:, 1] / np.max(self.umap_features[:, 1])
            self.umap_pos = {i: tuple(pos) for i, pos in enumerate(self.umap_features)}

        self.apply_filter()

        # Set initial zoom scale factor
        self.scale_factor = 1.1  # This controls the zoom intensity per scroll

        # Initialize pan variables
        self._panning = False
        self._pan_start = (0, 0)
        self.update_timer = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        
        # Initialize the canvas before calling redraw
        self.canvasG = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvasG.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        # self.canvasG.pack(side=TOP, fill=BOTH, expand=TRUE)
        # Draw initial hypergraph


        self.redraw_hypergraph()  # Now this can safely use self.canvas
        
        # Draw the initial canvas content
        self.canvasG.draw()
        self.canvasG.mpl_connect("button_press_event", self.select_in_hypergraph)
        # Bind events for zooming and panning
        self.canvasG.get_tk_widget().bind("<MouseWheel>", self.zoom)
        self.canvasG.get_tk_widget().bind("<ButtonPress-3>", self.start_pan)
        self.canvasG.get_tk_widget().bind("<B3-Motion>", self.pan)
        self.canvasG.get_tk_widget().bind("<ButtonRelease-3>", self.end_pan)

        
    def get_allowed_indices_from_filter(self):
        # If user has selected "exclude" edges -> create excluded_scenes, etc.
        # Or "include" edges -> create included_scenes.
        # Then gather the union of images from the resulting scenes.
        if self.chosen_edges.index("end") == 0:
            allowed_indices = np.arange(0,len(self.umap_features))
            return allowed_indices

        chosen_edges = self.chosen_edges.get(0,"end")
        
        if self.filter_mode.get() == "exclude":
            allowed_indices = [
                img
                for edge_name, images in self.hyperedges.items()                    
                if edge_name not in chosen_edges
                for img in images
            ]


        else:
            allowed_indices = [
                img
                for edge_name in chosen_edges
                for img in self.hyperedges[edge_name]
            ]
        return allowed_indices

    def start_pan(self, event):
        """Begin panning with right mouse button."""
        if event.num == 3:  # right-click
            self._panning = True
            self._pan_start = (event.x, event.y)
            # self.reset_timer()

    def pan(self, event):
        if not self._panning:
            return

        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)

        dx_data, dy_data = self.ax.transData.inverted().transform((dx, dy)) - \
                            self.ax.transData.inverted().transform((0, 0))
        dy_data = -dy_data

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim(xlim[0] - dx_data, xlim[1] - dx_data)
        self.ax.set_ylim(ylim[0] - dy_data, ylim[1] - dy_data)

        self.canvasG.draw()
        # self.reset_timer()

    def end_pan(self, event):
        self._panning = False


    def zoom(self, event):
        if self.ax is None or self.canvasG is None:
            return
        try:
            # Invert pixel coordinates to data coordinates.
            xdata, ydata = self.ax.transData.inverted().transform((event.x, event.y))
            scale_factor = 1 / self.scale_factor if event.delta > 0 else self.scale_factor

            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()

            new_xlim = [
                xdata - (xdata - xlim[0]) * scale_factor,
                xdata + (xlim[1] - xdata) * scale_factor
            ]
            new_ylim = [
                ydata - (ydata - ylim[0]) * scale_factor,
                ydata + (ylim[1] - ydata) * scale_factor
            ]

            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.canvasG.draw()
        except Exception as e:
            print("Zoom draw error:", e)
        # self.reset_timer()



    
    def reset_timer(self):
        if self.update_timer:
            self.update_timer.cancel()
        self.update_timer = Timer(0.5)#, self.update_visible_points)
        self.update_timer.start()
    
    def farthest_point_sampling(self, umap_pos, n_samples, initial_samples=None):
            """
            Selects `n_samples` points from `umap_pos` to be as evenly spread as possible using 
            the farthest-point sampling approach.
            
            Parameters:
                umap_pos (numpy.ndarray): 2D array with shape (n_points, 2) representing UMAP positions.
                n_samples (int): The number of samples to select.
                initial_samples (list): Optional list of initial points to include in the sampling.
                
            Returns:
                numpy.ndarray: Indices of selected samples, with shape (n_samples,).
            """

            umap_pos = np.array(umap_pos)
            sample_indices = []
            
            # Start with initial_samples if provided
            if initial_samples:
                sample_indices = list(range(len(initial_samples)))
            else:
                # Start with a random point if no initial samples
                sample_indices = [np.random.randint(umap_pos.shape[0])]

            # Iteratively add points that are farthest from the current sample
            for _ in range(len(sample_indices), n_samples):
                # Calculate the minimum distance from the existing samples to all points
                distances = np.min(
                    np.linalg.norm(umap_pos - umap_pos[sample_indices][:, np.newaxis], axis=2), 
                    axis=0
                )
                
                # Select the point farthest from the current set of sampled points
                next_index = np.argmax(distances)
                sample_indices.append(next_index)
            
            return np.array(sample_indices)
    
    def update_visible_points(self):
        # Figure out which points are in the current view
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        if not hasattr(self, 'previous_xlim'):
            self.previous_xlim = xlim
            self.previous_ylim = ylim

        # Determine if this is a zoom-out or not
        current_width = xlim[1] - xlim[0]
        previous_width = self.previous_xlim[1] - self.previous_xlim[0]
        current_height = ylim[1] - ylim[0]
        previous_height = self.previous_ylim[1] - self.previous_ylim[0]

        zoom_out = (
            (current_width > previous_width) or
            (current_height > previous_height)
        )

        # Just an example: we always pick points that are within the bounding box.
        visible_points = [
            i for i, (x, y) in enumerate(self.filtered_umap_features)
            if (xlim[0] <= x <= xlim[1]) and (ylim[0] <= y <= ylim[1])
        ]
        self.previous_xlim = xlim
        self.previous_ylim = ylim

        if not hasattr(self, 'previous_visible_points'):
            self.previous_visible_points = []

            # Maybe keep everything for zoom‐in
        self.previous_visible_points = visible_points

        # Now forcibly re‐draw
        self.redraw_hypergraph(update=True)


    # def update_visible_points(self):
    #     # Get the current visible area
    #     xlim = self.ax.get_xlim()
    #     ylim = self.ax.get_ylim()
        
    #     if not hasattr(self, 'previous_xlim'):
    #         self.previous_xlim = xlim
    #         self.previous_ylim = ylim
    #     # Determine if this is a zoom-out by comparing with previous limits
    #     current_width = xlim[1] - xlim[0]
    #     previous_width = self.previous_xlim[1] - self.previous_xlim[0]
    #     current_height = ylim[1] - ylim[0]
    #     previous_height = self.previous_ylim[1] - self.previous_ylim[0]

    #     zoom_out = (
    #         hasattr(self, 'previous_xlim') and
    #         (current_width > previous_width or current_height > previous_height)
    #     )


    #     if zoom_out:
    #         visible_points = [
    #             i for i, (x, y) in enumerate(self.filtered_umap_features)
    #             if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]
    #         ]
            
    #         visible_points = np.array(visible_points)
    #         vis_pts_umap = self.filtered_umap_features[visible_points]
    #         n_samples = min(25, len(vis_pts_umap))
    #         sampled_items = self.farthest_point_sampling(vis_pts_umap, n_samples=n_samples)

    #         sampled_indices = visible_points[sampled_items]
    #         # Step 2: Count how many of the new samples fall within the old bounding box
    #         within_old_box = [
    #             i for i in sampled_indices
    #             if (self.previous_xlim[0] <= self.filtered_umap_features[i][0] <= self.previous_xlim[1] and
    #                 self.previous_ylim[0] <= self.filtered_umap_features[i][1] <= self.previous_ylim[1])
    #         ]
    #         count_within_old_box = len(within_old_box)
    #         # Step 3: Choose that many points from previous_visible_points
    #         retained_points = self.previous_visible_points[:count_within_old_box]

    #         # Step 4: Add the remaining points from the new sampled_indices outside the old bounding box
    #         additional_points = [
    #             i for i in sampled_indices
    #             if i not in within_old_box
    #         ][:25 - len(retained_points)]

    #         # Final sample combining old and new points
    #         sample_indices = list(retained_points) + list(additional_points)

    #     else:
    #         # For zoom-in or regular behavior, keep standard sampling
    #         visible_points = [
    #             i for i, (x, y) in enumerate(self.filtered_umap_features)
    #             if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]
    #         ]

    #         # Retain points from the previous selection that are still visible
    #         retained_points = [i for i in self.previous_visible_points if i in visible_points]
    #         # Calculate the number of additional points needed
    #         points_needed = max(0, 25 - len(retained_points))

    #         # Sample additional points from the newly visible ones
    #         new_visible_points = [i for i in visible_points if i not in retained_points]
    #         if len(new_visible_points) <= points_needed:
    #             additional_points = new_visible_points
    #         else:
    #             sampled_indices = self.farthest_point_sampling(
    #                 [self.filtered_umap_features[i] for i in new_visible_points],
    #                 n_samples=points_needed
    #             )
    #             additional_points = [new_visible_points[i] for i in sampled_indices]
            

    #         sample_indices = list(retained_points) + list(additional_points)


    #     # Store current limits as previous for the next call
    #     self.previous_xlim = xlim
    #     self.previous_ylim = ylim
    #     # sample_indices = self.all_filtered_images[sample_indices]
        
    #     # Update the previous visible points for the next zoom event
    #     self.previous_visible_points = sample_indices  ## probably need to make sure it is the correct indices...from self.all_images or something
    #     # Create a new hypergraph and redraw
    #     self.scenes = self.create_scenes_from_samples(sample_indices, self.image_mapping)
    #     # scenes = self.create_scenes_from_samples(sampled_indices, self.image_mapping)

    #     self.redraw_hypergraph()

    def calculate_image_sizes(self, points, base_size=200):
        # Fit nearest neighbors in data coordinates
        nbrs = NearestNeighbors(n_neighbors=2).fit(points)
        distances, _ = nbrs.kneighbors(points)
        nearest_distances = distances[:, 1]  # Second column for nearest neighbor distance

        # Get current plot limits for scaling
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Calculate the data range in x and y directions
        data_width = xlim[1] - xlim[0]
        data_height = ylim[1] - ylim[0]

        # Get the figure size in inches and DPI for display scaling
        fig_width, fig_height = self.fig.get_size_inches() * self.fig.dpi

        # Convert data distances to display distances
        display_distances = nearest_distances / np.sqrt((data_width / fig_width) ** 2 + (data_height / fig_height) ** 2)

        # Scale image sizes and clip to defined range
        image_sizes = np.clip(display_distances / 2, 40, base_size)  # Adjust as needed

        return image_sizes

    def draw_umap_circles(self):
        """
        Draw each UMAP point as a Circle patch in data coords,
        so they scale automatically when zooming/panning.
        """

        # Optionally remove old circles if you want to re-draw
        if hasattr(self, 'circles'):
            for c in self.circles:
                c.remove()
        self.circles = []

        # For demonstration, let's pick a data-based radius like 0.01
        # (If your UMAP points typically fall in [0..1], that might be okay.)
        radius_in_data = 0.01

        for (px, py) in self.umap_features:
            circle = plt.Circle(
                (px, py),
                radius=radius_in_data,
                color='black',
                alpha=0.3,
                fill=True
            )
            self.ax.add_patch(circle)
            self.circles.append(circle)  

    def get_edge_color(self, e_name):
        if e_name not in self.edge_color_map:
            self.edge_color_map[e_name] = (random.random(), random.random(), random.random())
        return self.edge_color_map[e_name]

    def show_edge_view(self):
        """
        Display a hypergraph view that:
        - Shows the entire UMAP distribution in the background
        - Draws polygons for ONLY the selected edge + its overlapping edges
        - Adds interactive zoom & pan
        - Annotates images, keeping each thumbnail a constant screen size
        """
        
        cmap = plt.get_cmap('tab20')
        self.edge_color_map = {}

        for i, e_name in enumerate(self.hyperedges):
            # i % cmap.N handles the case where you have more than 20 edges.
            self.edge_color_map [e_name] = cmap(i % cmap.N)

        # 1) Bail out if no selected edge
        if not self.selected_edge:
            return

        # 2) Build a dict with the selected edge + overlap edges, skipping empty edges
        self.edges_to_draw = [self.selected_edge] + list(self.overlapping_hyperedge_names)
        self.sub_scenes = {}
        for e_name in self.edges_to_draw:
            if e_name in self.hyperedges:
                members = self.hyperedges[e_name]
                if len(members) > 0:
                    self.sub_scenes[e_name] = members

        if not self.sub_scenes:
            return  # nothing to show

        # # 3) If there's already a figure/canvas, destroy it
        if hasattr(self, 'canvasG') and self.canvasG is not None:
            self.canvasG.get_tk_widget().destroy()
            self.canvasG = None
        
        if hasattr(self, 'toolbar') and self.toolbar is not None:
            self.toolbar.destroy()
            self.toolbar = None


        # Create a fresh figure and axes
        # self.fig, self.ax = plt.subplots(figsize=(5,4))
        # self.ax.set_facecolor("#555555")
        # self.fig.set_facecolor("#555555")

        # 4) Ensure we have UMAP
        if len(self.umap_features) < 1:
            modelu = umap.UMAP()
            self.umap_features = modelu.fit_transform(self.features)
            # Normalize to [0,1]
            self.umap_features[:,0] -= self.umap_features[:,0].min()
            self.umap_features[:,1] -= self.umap_features[:,1].min()
            self.umap_features[:,0] /= self.umap_features[:,0].max()
            self.umap_features[:,1] /= self.umap_features[:,1].max()
            self.umap_pos = {
                i : tuple(self.umap_features[i])
                for i in range(len(self.umap_features))
            }
        self.apply_filter()

        self.scale_factor = 1.1  
        self._panning = False
        self._pan_start = (0, 0)
        self.update_timer = None


        # (1) Create a fresh figure and axes
        self.fig = Figure(figsize=(5,4))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#555555")
        self.fig.set_facecolor("#555555")

        # (2) Embed the Figure into a Canvas
        self.canvasG = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvasG.draw()
        self.canvasG.get_tk_widget().pack(side='top', fill='both', expand=True)

        # (3) Create and place the NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvasG, self.right_frame)
        self.toolbar.update()
        # If you want the toolbar at the top, use side='top'; 
        # or place it at the bottom with side='bottom'
        self.toolbar.pack(side='top', fill='x')

        # (4) Draw your hypergraph
        self.redraw_hypergraph()

        # (5) Finally, ensure the canvas is updated:
        self.canvasG.draw()






        # self.cid_press = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        # self.cid_release = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        # self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)


        # self.canvasG.get_tk_widget().bind("<MouseWheel>", self.zoom)
        # self.canvasG.get_tk_widget().bind("<ButtonPress-3>", self.start_pan)
        # self.canvasG.get_tk_widget().bind("<B3-Motion>", self.pan)
        # self.canvasG.get_tk_widget().bind("<ButtonRelease-3>", self.end_pan)




    def images_to_display(self):
        edges_to_draw = [self.selected_edge] if self.selected_edge else []
        for edge_id, var in self.checkbox_vars.items():
            if var.get():
                edges_to_draw.append(edge_id)
        sub_scenes = {}
        selected_images = set(self.hyperedges.get(self.selected_edge, []))
        for e_name in edges_to_draw:
            if e_name not in self.hyperedges:
                continue
            current_images = set(self.hyperedges[e_name])
            # Exclude images that appear in other drawn hyperedges.
            other_images = set()
            for other in edges_to_draw:
                if other != e_name and other in self.hyperedges:
                    other_images |= set(self.hyperedges[other])
            unique_images = list(current_images - other_images)
            if e_name == self.selected_edge:
                shared_images = list(selected_images - set(unique_images))
            else:
                shared_images = list((set(self.hyperedges[e_name]) & selected_images) - set(unique_images))
            combined = unique_images[:3] + shared_images[:3]
            sub_scenes[e_name] = tuple(combined)
        return sub_scenes


    

    def redraw_hypergraph(self, update=False):
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        self.ax.cla()
        self.ax.set_facecolor("#555555")
        self.fig.set_facecolor("#555555")
        self.ax.set_xlim(current_xlim)
        self.ax.set_ylim(current_ylim)
        self.ax.axis("on")
        self.ax.grid(True, which='both')


        if len(self.ax.collections) > 3:
            self.ax.collections[0].remove()
            self.ax.collections[0].remove()
            self.ax.collections[0].remove()
        # for coll in self.ax.collections:
        #     coll.set_facecolor("lightgrey")
        #     coll.set_edgecolor("lightgrey")
        for _ in range(len(self.ax.texts)):
            self.ax.texts[0].remove()

        if not self.selected_edge:
            return
        if not update:
            self.sub_scenes = self.images_to_display()
        if not self.sub_scenes:
            return
        if len(self.umap_features) < 1:
            import umap
            modelu = umap.UMAP()
            self.umap_features = modelu.fit_transform(self.features)
            self.umap_features[:, 0] -= self.umap_features[:, 0].min()
            self.umap_features[:, 1] -= self.umap_features[:, 1].min()
            self.umap_features[:, 0] /= self.umap_features[:, 0].max()
            self.umap_features[:, 1] /= self.umap_features[:, 1].max()
            self.umap_pos = {i: tuple(self.umap_features[i]) for i in range(len(self.umap_features))}
        self.apply_filter()

        # Split hyperedges: non-selected vs. selected.
        non_selected_scenes = {e: self.sub_scenes[e] for e in self.sub_scenes if e != self.selected_edge}
        selected_scene = {self.selected_edge: self.sub_scenes[self.selected_edge]} if self.selected_edge in self.sub_scenes else {}

        # Draw non-selected hyperedges.
        if non_selected_scenes:
            init_coll = len(self.ax.collections)
            hnx.drawing.draw(
                hnx.Hypergraph(non_selected_scenes),
                pos=self.umap_pos,
                ax=self.ax,
                with_node_labels=False,
                with_edge_labels=True,
                nodes_kwargs={"facecolors": "none", "edgecolors": "none", "alpha": 0, "linewidth": 0},
                edges_kwargs={
                    "facecolors": [to_rgba(self.get_edge_color(e), alpha=0.2) for e in non_selected_scenes.keys()],
                    "edgecolors": [self.get_edge_color(e) for e in non_selected_scenes.keys()],
                    "linewidth": [2 for _ in non_selected_scenes.keys()]
                }
            )
            for coll in self.ax.collections[init_coll:]:
                coll.set_zorder(100)

        # Draw selected hyperedge.
        if selected_scene:
            init_coll = len(self.ax.collections)
            hnx.drawing.draw(
                hnx.Hypergraph(selected_scene),
                pos=self.umap_pos,
                ax=self.ax,
                with_node_labels=False,
                with_edge_labels=True,
                nodes_kwargs={"facecolors": "none", "edgecolors": "none", "alpha": 0, "linewidth": 0},
                edges_kwargs={
                    "facecolors": [to_rgba(self.get_edge_color(self.selected_edge), alpha=0.2)],
                    "edgecolors": [self.get_edge_color(self.selected_edge)],
                    "linewidth": [6]
                }
            )
            for coll in self.ax.collections[init_coll:]:
                coll.set_zorder(200)

        # self.ax.set_facecolor("#555555")
        # self.fig.set_facecolor("#555555")
        # self.ax.set_xlim(current_xlim)
        # self.ax.set_ylim(current_ylim)
        # self.ax.axis("on")
        # self.ax.grid(True, which='both')
        # Push hyperedge labels on top.
        for txt in self.ax.texts:
            txt.set_zorder(1100)

        relevant_images = set()
        for e_name, img_ids in self.sub_scenes.items():
            relevant_images.update(img_ids)
        relevant_images = sorted(relevant_images)

        if not hasattr(self, 'image_boxes'):
            self.image_boxes = []
        else:
            for box in self.image_boxes:
                box.remove()
            self.image_boxes.clear()


        hyperedge_keys = list(self.sub_scenes.keys())
        other_edges = hyperedge_keys[1:]       # The remaining hyperedges.


        self.selected_indices_for_umap = set(self.hyperedges.get(self.selected_edge, []))
        for edge in other_edges:
            self.selected_indices_for_umap.update(self.hyperedges.get(edge, []))
        self.selected_indices_for_umap = sorted(self.selected_indices_for_umap)


        scatter_colors = []
        for img_id in self.selected_indices_for_umap:
            if img_id in self.hyperedges.get(self.selected_edge, []):
                scatter_colors.append(self.edge_color_map[self.selected_edge])
            else:
                # Check overlapping hyperedges.
                for edge in other_edges:
                    if img_id in self.hyperedges.get(edge, []):
                        scatter_colors.append(self.edge_color_map[edge])
                        break

        # Extract only the filtered points from self.filtered_umap_features.
        self.filtered_points = self.filtered_umap_features[self.selected_indices_for_umap]
        # Plot only the points in the selected indices.
        self.ax.scatter(
            self.filtered_points[:, 0],
            self.filtered_points[:, 1],
            c=scatter_colors,
            zorder=300
        )


        fixed_size = 50
        ## Draw context hyperedges (those not selected or overlapping) in light grey
        # context_edges = {
        #     edge: self.hyperedges[edge]
        #     for edge in self.hyperedges
        #     if edge not in ([self.selected_edge] + list(self.overlapping_hyperedge_names))
        #     and self.hyperedges[edge]
        # }
        # if context_edges:
        #     init_coll = len(self.ax.collections)
        #     hnx.drawing.draw(
        #         hnx.Hypergraph(context_edges),
        #         pos=self.umap_pos,
        #         ax=self.ax,
        #         with_node_labels=False,
        #         with_edge_labels=False,
        #         nodes_kwargs={
        #             "facecolors": "none",
        #             "edgecolors": "none",
        #             "alpha": 0,
        #             "linewidth": 0,
        #         },
        #         edges_kwargs={
        #             "facecolors": [to_rgba("lightgrey", alpha=0.1) for _ in context_edges.keys()],
        #             "edgecolors": ["lightgrey" for _ in context_edges.keys()],
        #             "linewidth": [1 for _ in context_edges.keys()]
        #         }
        #     )
            # Set a low zorder so they are drawn beneath the other hyperedges.
            # for coll in self.ax.collections[init_coll:]:
            #     coll.set_zorder(50)

        with h5py.File(self.hdf_path, 'r') as hdf:
            for img_id in relevant_images:
                x, y = self.umap_features[img_id]
                # Check original membership, not just what was drawn.
                if img_id in self.hyperedges.get(self.selected_edge, []):
                    border_color = self.edge_color_map[self.selected_edge]
                else:
                    border_color = None
                    for edge in self.overlapping_hyperedge_names:
                        if img_id in self.hyperedges.get(edge, []):
                            border_color = self.edge_color_map[edge]
                            break
                    if border_color is None:
                        border_color = "black"
                thumbnail_data = hdf['thumbnail_images'][img_id]
                pil_img = Image.fromarray(thumbnail_data.astype('uint8'))
                pil_img = pil_img.resize((fixed_size, fixed_size))
                offset_img = OffsetImage(pil_img, zoom=1)
                bboxprops = {"edgecolor": border_color, "linewidth": 2}
                if hasattr(self, 'selected_he_image_id') and img_id == self.selected_he_image_id:
                    bboxprops = {"edgecolor": "dodgerblue", "linewidth": 2}
                ab = AnnotationBbox(
                    offset_img,
                    (x, y),
                    xycoords="data",
                    boxcoords="offset points",
                    frameon=True,
                    bboxprops=bboxprops,
                    box_alignment=(0.5, 0.5),
                    zorder=1000
                )        
                ab.image_id = img_id  
                self.ax.add_artist(ab)
                self.image_boxes.append(ab)



        self.canvasG.draw()
        self.canvasG.get_tk_widget().bind("<MouseWheel>", self.zoom)
        self.canvasG.get_tk_widget().bind("<ButtonPress-3>", self.start_pan)
        self.canvasG.get_tk_widget().bind("<B3-Motion>", self.pan)
        self.canvasG.get_tk_widget().bind("<ButtonRelease-3>", self.end_pan)
        self.cid_press = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.cid_release = self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_button_press)
        for coll in self.ax.collections:
            coll.set_picker(True)


    def on_pick(self, event):
        """
        Handles pick events for selecting hyperedges or points in the hypergraph.

        Args:
            event: The pick event containing information about the clicked artist.
        """
        if (event.mouseevent is None 
            or event.mouseevent.button != 1  # left button is usually "1"
            or event.mouseevent.name not in ['button_press_event','button_release_event']):
            return
        picked_object = event.artist  # The artist object that was clicked
        self.selected_hyperedges = []
        if isinstance(picked_object, PolyCollection):  # Check for hyperedge areas
            click_pos = (event.mouseevent.xdata, event.mouseevent.ydata)
            edge_selected = False

            # Iterate through all paths in the PolyCollection
            for idx, path in enumerate(picked_object.get_paths()):
                # Check if the click is near the edge of the polygon (adjust radius for sensitivity)
                if path.contains_point(click_pos, radius=0.005):  # Click near edge
                    if not path.contains_point(click_pos):  # Exclude points inside the polygon
                        self.selection = idx  # Assign the hyperedge ID
                        edge_id = self.incidence_edge_ids[idx]
                        self.selected_hyperedges = [edge_id]  # Select only this edge
                        print(f"Exact edge {edge_id} selected.")
                        edge_selected = True
                        break  # Stop checking once an edge is selected

            # If no exact edge was clicked, check for interior clicks
            if not edge_selected:
                for idx, path in enumerate(picked_object.get_paths()):
                    if path.contains_point(click_pos):  # Click inside polygon
                        try:
                            edge_id = self.incidence_edge_ids[idx]
                            self.selected_hyperedges.append(edge_id)
                        except IndexError:
                            print('index error idx')
                            

                        
                print(f"Hyperedges {self.selected_hyperedges} selected.")

            # Force a redraw of the canvas
            event.canvas.draw_idle()

    def handle_double_click(self, event):
        # if clicking image, 
        self.select_in_hypergraph(event)
        # if clicking inside edge, display union of overlapping hyperedges. 

    

    def on_button_press(self, event):
        print('im the captain now')
        if event.inaxes != self.ax:
            return
        print('wayo')
        shift_held = (event.key == 'shift')
        self.control_held = (event.key == 'control')
        # Double left-click => do double-click logic
        if event.dblclick and event.button == 1:
            print('wayo2')
            self.handle_double_click(event)
            return
        
        # SHIFT + right-click => get_img_idx_from_event
        if shift_held and event.button == 2:
            print('thiosone')
            self.get_img_idx_from_event(event)
            return
        
        # Normal right-click => select_in_hypergraph
        if event.button == 2:
            self.select_in_hypergraph(event)
            return

        if event.button == 1 and self.control_held and self.lasso is None:
            self.lasso = LassoSelector(self.ax, onselect=self.on_lasso_select,
                                       props={'color':'red','linewidth':1,'alpha':0.8})

    def on_key_release(self, event):
        """Called whenever a key is released over the figure."""
        if event.key == 'control':
            self.control_held = False
            # If a lasso was in progress, disconnect it
            if self.lasso:
                self.lasso.disconnect_events()
                self.lasso = None

            

    def select_in_hypergraph(self, event):
        # Check if the left mouse button was clicked (button 1)
        if event.button != 1:
            return

        click_x, click_y = event.x, event.y
        data_x, data_y = self.ax.transData.inverted().transform((click_x, click_y))

        for image_box in self.image_boxes:
            bbox = image_box.get_window_extent(self.canvasG.get_renderer())
            bbox_data = bbox.transformed(self.ax.transData.inverted())
            if bbox_data.contains(data_x, data_y):
                self.selected_image_id = image_box.image_id  # Retrieve the stored image id
                print(f"Image selected: {self.selected_image_id}")
                # Continue with further processing (e.g., update the overlapping hyperedges view)
                self.display_overlapping_hyperedges()
                self.after(100, self.scroll_to_selected_image)
                return





    def scroll_to_selected_image(self):
        # Make sure we have a selected image
        if not hasattr(self, 'selected_image_id'):
            return
        selected_img = self.selected_image_id

        # First, try to find the selected image in the overlapping hyperedges
        target_index = None
        for he_index, hyperedge in enumerate(self.overlapping_hyperedges):
            # hyperedge is a NumPy array or list of image ids
            if selected_img in hyperedge:
                target_index = he_index
                break

        if target_index is not None:
            # Scroll the main canvas (for overlapping hyperedges) as before.
            target_canvas = self.hyperedge_canvases[target_index]
            target_frame = self.hyperedge_frames[target_index]
            self.bottomleft_imex_frame.update_idletasks()
            self.main_canvas.update_idletasks()
            # Get the y-coordinate of the target frame
            y_coord = target_frame.winfo_y()
            bbox_all = self.main_canvas.bbox("all")
            total_height = bbox_all[3] - bbox_all[1]
            if total_height > 0:
                fraction = max(0, min(y_coord / total_height, 1))
                self.main_canvas.yview_moveto(fraction)
            # Now scroll inside the target_canvas if a "selected_tag" exists.
            bbox_selected = target_canvas.bbox('selected_tag')
            if bbox_selected:
                x1, y1, x2, y2 = bbox_selected
                scrollregion = target_canvas.cget("scrollregion")
                if scrollregion:
                    _, _, _, y_max = scrollregion.split()
                    y_max = float(y_max)
                    if y_max > 0:
                        fraction_img = y1 / y_max
                        target_canvas.yview_moveto(fraction_img)
        else:
            # The selected image is not in the overlapping hyperedges.
            # Check if it is in the selected hyperedge.
            selected_images = set(self.hyperedges.get(self.selected_edge, []))
            if selected_img in selected_images:
                # Assume self.c is the canvas displaying the selected hyperedge images.
                self.c.update_idletasks()
                # Find the bounding box for the image with tag equal to selected_img.
                selected_tag = f"img_{selected_img}"
                items = self.c.find_withtag(selected_tag)
                if items:
                    bbox_img = self.c.bbox(items[0])
                if bbox_img:
                    x1, y1, x2, y2 = bbox_img
                    scrollregion = self.c.cget("scrollregion")
                    if scrollregion:
                        # scrollregion is a string like "0 0 width height"
                        _, _, _, y_max = scrollregion.split()
                        y_max = float(y_max)
                        if y_max > 0:
                            fraction_img = y1 / y_max
                            self.c.yview_moveto(fraction_img)
                    # Remove any previous highlight (if you use a dedicated tag)
                    self.c.delete("selected_tag")
                    # Draw a rectangle around the selected image to highlight it.
                    self.c.create_rectangle(x1, y1, x2, y2, outline="dodgerblue", width=2, tags="selected_tag")
                    self.c.update_idletasks()
            else:
                return
                                            
    def click_select_in_overlapping_hyperedges(self, event):
        """
        Right-click handler for the overlapping hyperedge canvas.
        """
        if event.num == 3:  # Right-click
            canvas = event.widget
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            # Find the clicked item on the canvas
            clicked_item = canvas.find_closest(x, y)
            if clicked_item:
                tags = canvas.gettags(clicked_item)
                image_idx = None
                for t in tags:
                    if t.isdigit():
                        image_idx = int(t)
                        break
                if image_idx is not None:
                    # Determine which hyperedge this canvas corresponds to
                    hyperedge_index = self.hyperedge_canvases.index(canvas)
                    
                    # Linked mode
                    if self.linked.var.get():
                        # Scenes is presumably a dict: {hyperedge_name: set_of_images, ...}
                        # We can get the hyperedge name directly:
                        hyperedge_name = list(self.scenes.keys())[hyperedge_index]
                    else:
                        # Non-linked mode: using self.edge_ids, which should store the actual hyperedge names
                        hyperedge_name = self.edge_ids[hyperedge_index]

                    # Then call your focus function with the actual hyperedge name
                    self.focus_on_image_in_projection(image_idx, hyperedge_name)

    def click_select_in_matrix(self, event):
        """
        Right-click handler for the overlapping hyperedge canvas.
        """
        if event.num == 3:  # Right-click
            canvas = event.widget
            x = canvas.canvasx(event.x)
            y = canvas.canvasy(event.y)
            # Find the clicked item on the canvas
            clicked_item = canvas.find_closest(x, y)
            if clicked_item:
                tags = canvas.gettags(clicked_item)
                image_idx = None
                for t in tags:
                    if t.isdigit():
                        image_idx = int(t)
                        break
                if image_idx is not None:
                    if len(self.umap_features) > 0:
                        self.focus_on_image_in_projection(image_idx, self.selected_edge)



    def focus_on_image_in_projection(self, image_idx, hyperedge_key):
        """
        1. Zoom the UMAP projection so the entire hyperedge is in view.
        2. From the now visible points in the projection, pick the clicked image plus 24 more using farthest-point sampling.
        3. Redraw the hypergraph with these 25 points.
        """

        # Get all images of that hyperedge
        hyperedge_images = self.filtered_hyperedges[hyperedge_key]
        hyperedge_images = list(map(int, hyperedge_images))

        # Compute bounding box of hyperedge images in the UMAP projection
        hyperedge_positions = self.umap_features[hyperedge_images]
        x_coords = hyperedge_positions[:, 0]
        y_coords = hyperedge_positions[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)

        # Add padding so they're not at the very edge
        padding_factor = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= x_range * padding_factor
        x_max += x_range * padding_factor
        y_min -= y_range * padding_factor
        y_max += y_range * padding_factor

        # Set the axes to show the entire hyperedge
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.canvasG.draw_idle()
        self.bottomleft_imex_frame.update_idletasks()
        self.main_canvas.update_idletasks()

        # Now, determine which points are visible in this new view:
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Find visible points based on current axes
        visible_points = [
            i for i, (x, y) in enumerate(self.filtered_umap_features)
            if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]
        ]

        # Make sure the clicked image is visible (it should be since we zoomed to the hyperedge containing it)
        if image_idx not in visible_points:
            # If for some reason the image is not visible, we might need to adjust zoom or return.
            print("Clicked image not visible after zoom, adjusting...")
            # For simplicity, assume it is visible. If not, you may need to broaden the bounds and redraw.
            return

        visible_positions = self.filtered_umap_features[visible_points]
        selected_pos = [self.filtered_umap_features[image_idx]]  # initial sample

        sampled_relative = self.farthest_point_sampling(
            visible_positions, n_samples=25, initial_samples=selected_pos
        )
        sampled_indices = [visible_points[i] for i in sampled_relative]

        if image_idx not in sampled_indices:
            sampled_indices[0] = image_idx  # force it in

        self.previous_visible_points = sampled_indices
    
        self.selected_he_image_id = image_idx
        print(self.selected_he_image_id)
        # Create scenes from these samples
        self.scenes = self.create_scenes_from_samples(self.previous_visible_points, self.image_mapping)
        # Redraw hypergraph
        self.redraw_hypergraph(self.scenes)


    def add_to_chosen_edges(self):
        """
        Adds the currently selected hyperedges from self.categories
        into the second listbox self.chosen_edges (if not already present).
        """
        
        selections = self.categories.curselection()
        for sel in selections:
            edge_name = self.categories.get(sel)
            # Prevent duplicates
            if edge_name not in self.chosen_edges.get(0, END):
                self.chosen_edges.insert(END, edge_name)
        
        # Optionally apply the filter immediately after adding
        # self.apply_filter()


    def remove_from_chosen_edges(self):
        """
        Removes the currently selected hyperedges from self.chosen_edges.
        """
        selections = list(self.chosen_edges.curselection())
        # Because removing items changes indices, reverse the list so we remove from the bottom up
        selections.reverse()
        for sel in selections:
            self.chosen_edges.delete(sel)

        # Optionally apply the filter immediately after removing
        # self.apply_filter()

    def apply_filter(self):
        """
        Reads self.filter_mode (exclude/include) and the hyperedges in self.chosen_edges,
        then filters self.scenes accordingly and calls redraw_hypergraph.
        """
        if self.chosen_edges.index("end") == 0:
            self.filtered_hyperedges = self.hyperedges
            self.filtered_umap_features = self.umap_features
            all_imgs = set()
            for images in self.filtered_hyperedges.values():
                all_imgs.update(images)
            self.all_filtered_images = np.array(sorted(all_imgs))
            return
        
        chosen_edges = self.chosen_edges.get(0,"end")
        mode = self.filter_mode.get()

        if mode == "exclude":
            self.filtered_hyperedges = {
                edge_name: images
                for edge_name, images in self.hyperedges.items()
                if edge_name not in chosen_edges
            }
        else:
            # mode == "include"
            self.filtered_hyperedges = {
                edge_name: self.hyperedges[edge_name]
                for edge_name in chosen_edges
                if edge_name in self.hyperedges
            }

        # self.all_filtered_images = np.array([img for images in self.filtered_hyperedges.values() for img in images])   
        all_imgs = set()
        for images in self.filtered_hyperedges.values():
            all_imgs.update(images)
        self.all_filtered_images = np.array(sorted(all_imgs))

        self.filtered_umap_features = self.umap_features[self.all_filtered_images]


    def exclude_scenes(self, scenes_dict, edges_to_exclude):
        """
        Returns a new scenes dictionary that excludes the given hyperedges,
        and also excludes images that are present *only* in those hyperedges.
        """
        excluded_set = set(edges_to_exclude)
        filtered_scenes = {}

        for he_name, images in scenes_dict.items():
            # If this hyperedge is excluded, skip it
            if he_name in excluded_set:
                continue

            # Otherwise, keep it but filter out images that belong *only* to excluded edges
            kept_images = []
            for img in images:
                # Find all hyperedges that contain this image
                edges_for_img = set(self.image_mapping[img])

                # If edges_for_img is entirely within excluded_set,
                # that means this image has nowhere else to appear -> exclude it.
                if edges_for_img.issubset(excluded_set):
                    # discard this image
                    continue
                else:
                    kept_images.append(img)

            # Keep the hyperedge if there's at least one image left
            if kept_images:
                filtered_scenes[he_name] = tuple(kept_images)

        return filtered_scenes

    def include_scenes(self, scenes_dict, edges_to_include):
        """
        Returns a new scenes dictionary that ONLY includes the given hyperedges,
        and by definition, only includes images in those edges.
        """
        included_set = set(edges_to_include)
        filtered_scenes = {}

        for he_name, images in scenes_dict.items():
            # Keep only the edges that are explicitly in included_set
            if he_name in included_set:
                filtered_scenes[he_name] = images

        return filtered_scenes



    #function to focus on a selected image. From here you can draw a square to select a part of an image to compare against all other images                   
    def focus_image(self):
        if len(self.selected_images) == 1:
            self.c['scrollregion'] = (0,0,0,800)
            self.squares = None
            im_tag = self.c.gettags(self.imagex[self.selected_images[0]])
            self.bucketDisp = 5
            self.c.delete("all")
            self.focused_image = []
            d = self.im_list[int(im_tag[0])]
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'focused image')
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(int(im_tag[0])))
            
            self.focused_image.append(d)
            for j in range(0,len(self.focused_image)):
                load = Image.open(self.focused_image[j])
                try:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation]=='Orientation':
                            break
                    exif=dict(load._getexif().items())
                
                    if exif[orientation] == 3:
                        load=load.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        load=load.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        load=load.rotate(90, expand=True)                
                except AttributeError:
                    pass
                except KeyError:
                    pass

                load = load.resize((800,800))
                render = ImageTk.PhotoImage(load)
            # labels can be text or images
                my_img = Label(self,background='#555555')#, image=render)
                my_img.image = render
                #my_img.grid()
                #image_.append(my_img)
                row_ = math.floor(j/self.num_im_row)
                column_ = j%self.num_im_row
                #image_[j].grid(row = row_, column = column_)
                self.imagex.append(self.c.create_image(column_*(800 + self.image_distance)+ (800 / 2),row_*(800 + self.image_distance)+ (800 / 2), image =render,tags=im_tag))

    def rank_image(self):
        if len(self.selected_images) == 1:
            #self.c['scrollregion'] = (0,0,0,600)
            #self.squares = None
            im_tag = self.c.gettags(self.imagex[self.selected_images[0]])
            self.bucketDisp = 2
    
            im_tag = int(float(im_tag[0]))
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'query image')
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(im_tag))
            
            with h5py.File(self.hdf_path, 'r') as hdf:
                self.rank_list = np.array(hdf.get('cm')[int(im_tag),:])            
            #self.rank_list = self.cm[:,int(im_tag)] # get all the correlations between the selected image and all the other images
            if len(self.video_cm)>0:
                self.rank_list = np.hstack((self.rank_list,self.video_cm[int(im_tag),:]))
            temp_list = np.sort(self.rank_list,0)[::-1] #sorts the correlations
            self.rank_list = np.argsort(self.rank_list,0)[::-1] #sorts the id of all the images based on correlation
            self.c.xview_moveto(self.origX)  #####
            self.c.yview_moveto(self.origY) ######
            self.rank_list = np.asarray(self.rank_list)
            temp_list = np.asarray(temp_list)
            temp_list[np.isnan(temp_list)] = -100   #some images have no correlations due to issues with loading and extracting the image features. Ususally means the image file is damaged
            self.rank_list = self.rank_list[temp_list>-50] # this removes all the broken images
            self.rank_list = np.append(im_tag,self.rank_list) # this adds the selected (queried) image to the image list to be displayed
            #self.bucketDisp = 0
            self.c.delete("all")
            self.display_images(self.rank_list) #function to display the ranked list of images. By default it displays all images, but may need to be limited for large datasets

            self.communication_label['background'] = '#FFFFFF'
            self.communication_label.configure(text='Finished calculating. Showing the ranking')
            #plt.close("all")

    def rank_bucket(self):
        if self.bucketDisp == 1:
            self.bucketDisp = 0
            cur_buck = np.sort(self.current_bucket)
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'rank bucket')
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(cur_buck))
            with h5py.File(self.hdf_path, 'r') as hdf:
                cmlen = hdf.get('cm').len()
                cur_buck = cur_buck[cur_buck<cmlen]
                bucket_cm = np.array(hdf.get('cm')[cur_buck])
            
            #bucket_cm = self.cm[self.current_bucket]
            
            bucket_cm2 = bucket_cm.reshape((-1,1))
            super_index = np.arange(0,len(self.im_list))
            super_index = np.resize(super_index,len(super_index)*len(cur_buck))
            #super_index = super_index[np.flip(np.argsort(np.nan_to_num(bucket_cm2),0))]
            super_index = pd.unique(super_index[np.flip(np.argsort(np.nan_to_num(bucket_cm2),0))].squeeze())
            
            self.c.xview_moveto(self.origX)  #####
            self.c.yview_moveto(self.origY) ######
            self.rank_list = super_index
            self.c.delete("all")
            self.display_images(self.rank_list) #function to display the ranked list of images. By default it displays all images, but may need to be limited for large datasets

            self.communication_label['background'] = '#FFFFFF'
            self.communication_label.configure(text='Finished calculating. Showing the ranking')

    def source_cluster(self):
        try:
            if len(self.selected_images) == 1:
                #self.c['scrollregion'] = (0,0,0,600)
                #self.squares = None
                im_tag = self.c.gettags(self.imagex[self.selected_images[0]])
                self.bucketDisp = 2
        
                im_tag = int(float(im_tag[0]))            
                
                #s_cluster = self.df[self.df.isin([im_tag])].stack().keys()[0][1]
                s_cluster = np.where(self.df == im_tag)[1]
                self.bucketDisp = 0
                self.imagex = []
                self.c.delete("all")
                num_c = s_cluster
                #df = pd.read_csv('forAppOID.csv', header=None) 
                #df=df-1 #
                self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'source cluster')                
                self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'cluster number' + str(num_c))

                try:
                    cluster = self.df[:,num_c]
                    cluster = cluster[cluster > -1]
                    self.currentcluster = cluster
                    self.greentangles = []
                    self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters. This cluster contains ' + str(len(cluster)) + ' images.')
                except KeyError:
                    cluster = []         
                self.display_images(cluster)
        except TypeError:
            im_tag = self.selected_images
            #s_cluster = self.df[self.df.isin([im_tag])].stack().keys()[0][1]
            s_cluster = np.where(self.df == im_tag)[1]
            self.bucketDisp = 0
            self.imagex = []
            self.c.delete("all")
            num_c = s_cluster
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'source cluster')                
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'cluster number' + str(num_c))
            #df = pd.read_csv('forAppOID.csv', header=None) 
            #df=df-1 #
            try:
                cluster = self.df[:,num_c]
                cluster = cluster[cluster > -1]
                self.currentcluster = cluster
                self.greentangles = []
                self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters. This cluster contains ' + str(len(cluster)) + ' images.')
            except KeyError:
                cluster = []
            self.display_images(cluster)

    def display_cluster_info(self):
        try:
            if len(self.selected_images) == 1:
                im_tag = self.c.gettags(self.imagex[self.selected_images[0]])
                im_tag = int(float(im_tag[0]))
                s_cluster = np.where(self.df == im_tag)[1]
    
                # Print the cluster number
                cluster_num = s_cluster[0] if s_cluster.size > 0 else None
                if cluster_num is not None:
                    cluster_info = f'Image is in cluster number: {cluster_num}.'
    
                    # Check which buckets the image is in
                    buckets = [bucket for bucket, images in self.theBuckets.items() if im_tag in images]
                    if buckets:
                        bucket_info = f' The image is also in buckets: {", ".join(buckets)}.'
                    else:
                        bucket_info = ' The image is not in any buckets.'
                    
                    self.communication_label.configure(text=cluster_info + bucket_info)
                else:
                    self.communication_label.configure(text='Cluster number not found.')
        except TypeError:
            self.communication_label.configure(text='Error in processing the selected image.')                

    def vcorrcoef(X,y):
        Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
        ym = np.mean(y)
        r_num = np.sum((X-Xm)*(y-ym),axis=1)
        r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
        r = r_num/r_den
        return r



    def filter_buckets(self):
    
        def find_intersection(lists):
            # Start with the first list's set
            result = set(lists[0])
            # Intersect with the subsequent sets
            for lst in lists[1:]:
                result.intersection_update(lst)
            return list(result)
        
        self.cluster_in = []
        self.cluster_out = []
        
        # Gather the lists for filter_in
        filter_in_names = []
        for i in range(len(self.filter_in)):
            bucket_name = self.categories.get(self.filter_in[i])
            filter_in_names.append(bucket_name)
            self.cluster_in.append(self.theBuckets[bucket_name])
        
        # Find the intersection if there are multiple lists
        if len(self.filter_in) > 1:
            self.cluster_in = find_intersection(self.cluster_in)
        else:
            self.cluster_in = list(chain.from_iterable(self.cluster_in))
        
        # Initialize filter_out if not already done
        if not hasattr(self, 'filter_out'):
            self.filter_out = []
    
        # Gather the lists for filter_out
        filter_out_names = []
        for i in range(len(self.filter_out)):
            bucket_name = self.categories.get(self.filter_out[i])
            filter_out_names.append(bucket_name)
            self.cluster_out.append(self.theBuckets[bucket_name])
        
        # Flatten the cluster_out list
        self.cluster_out = list(chain.from_iterable(self.cluster_out)) 
    
        # Get the difference between cluster_in and cluster_out
        self.filtered_bucket = list(set(self.cluster_in) - set(self.cluster_out))
        filter_in_text = ', '.join(filter_in_names)
        filter_out_text = ', '.join(filter_out_names)
        self.communication_label.configure(text=f'Used buckets: {filter_in_text}. Excluded buckets: {filter_out_text}.')
        self.imagex = []
        self.c.delete("all")
        self.display_images(self.filtered_bucket)
        
        # Reset the filters
        self.filter_in = []
        self.filter_out = []

    def clear_filter(self):
        self.categories2.selection_clear(0, END)
        self.categories3.selection_clear(0, END)

        
    # This function compares the drawn square to all other images                   
    def query_selection(self):
        load = Image.open(self.focused_image[0])
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'query selected square')                
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(self.focused_image[0]))
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation]=='Orientation':
                    break
            exif=dict(load._getexif().items())
        
            if exif[orientation] == 3:
                load=load.rotate(180, expand=True)
            elif exif[orientation] == 6:
                load=load.rotate(270, expand=True)
            elif exif[orientation] == 8:
                load=load.rotate(90, expand=True)                
        except AttributeError:
            pass
        except KeyError:
            pass

        width, height = load.size
        evex1 = min(self.evex1,self.evex2)
        evex2 = max(self.evex1,self.evex2)
        evey1 = min(self.evey1,self.evey2)
        evey2 = max(self.evey1,self.evey2)
        
        evex1 = width / (800 / evex1)
        evex2 = width / (800 / evex2)
        evey1 = height / (800 / evey1)
        evey2 = height / (800 / evey2)
        

        def feature_extraction2(neural_net, im_list,evex1,evex2,evey1,evey2):
            f = im_list
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if neural_net == 'inception_v3':
                f_size = 2048
                model = models.inception_v3(pretrained='imagenet')
                layer = model._modules.get('Mixed_7c')
            elif neural_net == 'resnet152': #2084
                f_size = 2048
                model = models.resnet152(pretrained=True)
                layer = model._modules.get('avgpool')
            elif neural_net == 'resnet18': #512
                f_size =512
                model = models.resnet18(pretrained=True)
                layer = model._modules.get('avgpool')
            elif neural_net == 'vgg16': #4096
                f_size = 4096
                model = models.vgg16(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer   = 'nothing'
            elif neural_net == 'vgg19': #4096
                f_size =    4096
                model = models.vgg19(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            elif neural_net == 'densenet161': #2208
                f_size =2208
                model = models.densenet161(pretrained=True)	
                model = model.features
            elif neural_net == 'squeezenet1_0': #512
                f_size = 1000
                model = models.squeezenet1_0(pretrained=True)
                #model = model.features
                layer = 'nothing'
            elif neural_net == 'alexnet':    
                f_size = 4096
                model = models.alexnet(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            model.eval()
            model = model.to(device)
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            if neural_net == 'inception_v3':
                transform = transforms.Compose([
                            transforms.Resize((299,299)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            else:
                transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            def get_vector2(image_name,f_size, layer, transform, model, neural_net,evex1,evex2,evey1,evey2):
                img = Image.open(image_name)
                try:
                    for orientation in ExifTags.TAGS.keys():
                        if ExifTags.TAGS[orientation]=='Orientation':
                            break
                    exif=dict(img._getexif().items())
                
                    if exif[orientation] == 3:
                        img=img.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        img=img.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        img=img.rotate(90, expand=True)                
                except AttributeError:
                    pass
                except KeyError:
                    pass
                img = img.crop((evex1,evey1,evex2,evey2))
                
                if img.mode == 'RGB':
                    try:
                        t_img = transform(img).unsqueeze(0)
                    except OSError:
                        t_img = transform(img).unsqueeze(0)
                    t_img = t_img.to(device)
                    if neural_net == 'alexnet' or neural_net =='vgg19' or neural_net =='vgg16' or neural_net =='alexnet' or neural_net =='squeezenet1_0':
                        torch.cuda.empty_cache()
                        my_embeddingz = model(t_img)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    elif neural_net == 'densenet161':
                        featuresY = model(t_img)
                        my_embeddingz = F.relu(featuresY,inplace= True)
                        my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=7, stride=1).view(featuresY.size(0), -1)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    elif neural_net == 'inception_v3':
                        my_embeddingz = torch.zeros((1,f_size,8,8))
                        my_embeddingz = my_embeddingz.to(device)
                            # 4. Define a function that will copy the output of a layer
                        def copy_data(m, i, o):
                            my_embeddingz.copy_(o.data)
                            # 5. Attach that function to our selected layer
                        h = layer.register_forward_hook(copy_data)
                            # 6. Run the model on our transformed image
                        model(t_img)
                        #    # 7. Detach our copy function from the layer
                        h.remove()
                        my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=8)
                        my_embeddingz = my_embeddingz.view(my_embeddingz.size(0), -1)
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    else:
                        my_embeddingz = torch.zeros((1,f_size,1,1))
                        my_embeddingz = my_embeddingz.to(device)
                            # 4. Define a function that will copy the output of a layer
                        def copy_data(m, i, o):
                            my_embeddingz.copy_(o.data)
                            # 5. Attach that function to our selected layer
                        h = layer.register_forward_hook(copy_data)
                            # 6. Run the model on our transformed image
                        model(t_img)
                        #    # 7. Detach our copy function from the layer
                        h.remove()
                        my_embeddingz = my_embeddingz.cpu()
                        my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                else:
                    my_embeddingz = np.zeros((f_size,))
                return my_embeddingz
            my_embed = []
            self.progress["value"] = 0
            self.progress["maximum"] = len(f)-1
            for i in range(0,len(f)):
                my_embed.append(get_vector2(f[i],f_size,layer,transform,model,neural_net,evex1,evex2,evey1,evey2))

            features = np.asarray(my_embed)
            return features


        self.focusfeatures = feature_extraction2(self.neuralnet,self.focused_image,evex1,evex2,evey1,evey2)

        def create_matrix2(focusfeatures,features,distance_metric):
            focusfeatures = np.squeeze(np.expand_dims(focusfeatures,0))
            features = np.squeeze(np.expand_dims(features,0))
            features_t = np.transpose(features)
            focusfeatures_t = np.transpose(focusfeatures)
            cm = []                
            sumX = sum(features_t)
            focussumX = sum(focusfeatures_t)
            sumsquareX = sum(features_t**2)
            
#                for i in range(0,features.shape[0]):
            feat0 = focusfeatures
            sumXY = np.dot(feat0,features_t)
            r = features.shape[1]*sumXY - focussumX*sumX 
            s = ((features.shape[1] * sumsquareX) - sumX**2)
            t = 1./((s[0]*s)**0.5)
            u = r * t
            cm.append(u)
            cm = np.asarray(cm)
            return cm
        self.focuscm = create_matrix2(self.focusfeatures,self.features,'correlation')
        temp_list = np.sort(self.focuscm.transpose(),0)[::-1]
        
        self.focus_list = np.argsort(self.focuscm.transpose(),0)[::-1]
        #self.rank_list = self.rank_list[0:1000]
        self.c.xview_moveto(self.origX)  #####
        self.c.yview_moveto(self.origY) ######
        self.focus_list = np.asarray(self.focus_list)
        temp_list = np.asarray(temp_list)
        temp_list[np.isnan(temp_list)] = -100
        self.focus_list = self.focus_list[temp_list>-50]
        self.bucketDisp = 0
        self.c.delete("all")
        x = []
        for i in range(0,len(self.focus_list)):
            d = self.im_list[self.focus_list[i]]
            x.append(d)
        self.display_images(self.focus_list)

    #Clusters the currently displayed images into subclusters per the selected images by the user
    def subcluster(self):
        m = len(self.selected_images) #number of images selected (number of subclusters)
        im_tags = [] 
        for ts in range(len(self.imagex)):
            im_tags.append(int(self.c.gettags(self.imagex[ts])[0])) #all images currently displayed
        im_tags = np.asarray(im_tags)
        im_tags = np.sort(im_tags)
        self.sel_im = im_tags[self.selected_images] #id of images currently selected
        #sel_im = sel_im.to_numpy()
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'subclustering')
        with h5py.File(self.hdf_path, 'r') as hdf:
                cmlen = hdf.get('cm').len()
                im_tags = im_tags[im_tags<cmlen]
                cur_cm = np.array(hdf.get('cm')[im_tags])
        
        #cur_cm = self.cm[im_tags] #correlations of images currently selected
        
        cur_cm = cur_cm[:,self.sel_im] #correlations of images currently selected
        cur_cm = np.nan_to_num(cur_cm) #removes nans
        cur_cmax = np.argmax(cur_cm,1) #determine which selected image correlates higher with each image
        nco = np.array([],dtype='int') #empty array for subclustered images
        for kk in range(m): #going through the selected images
            tb = im_tags[cur_cmax==kk]
            tc = cur_cm[cur_cmax==kk][:,kk]
            
            tb = tb[np.flip(np.argsort(tc))]
            tb = np.delete(tb,np.intersect1d(tb,self.sel_im,return_indices=True)[1],0)
            try:
                #nco= np.hstack([nco,sel_im[kk],self.currentcluster[cur_cmax==kk]])
                nco= np.hstack([nco,self.sel_im[kk],tb])

            except ValueError:
                #nco= np.vstack([nco,sel_im[kk],self.currentcluster[cur_cmax==kk]])
                nco= np.vstack([nco,self.sel_im[kk],tb])
        #for overview:
        try:
            self.xsorted
        except AttributeError:
            self.xsorted = np.argsort(im_tags)
            self.xsortedimtags = im_tags[self.xsorted]
                            
        ypos = np.searchsorted(self.xsortedimtags, nco)
        
        self.ind_for_overview = self.xsorted[ypos]
        ###
        self.subcyes = 1
        self.display_images(nco)
        
    def subclusterkmeans(self):
        m = len(self.selected_images)
        im_tags = [] 
        for ts in range(len(self.imagex)):
            im_tags.append(int(self.c.gettags(self.imagex[ts])[0]))
        im_tags = np.asarray(im_tags)
        self.sel_im = im_tags[self.selected_images]
        
        
        def color_features(an_image):
            an_image = an_image.convert('HSV')    
            imarray = np.asarray(an_image)
            imarray = imarray.reshape(imarray.shape[0]*imarray.shape[1],3)
            reimarray = np.vstack((imarray[:,0]/255*360,imarray[:,1]/255,imarray[:,2]/255)).transpose()
            reimarray[:,0][reimarray[:,0] < 20] = 0
            reimarray[:,0][np.where((reimarray[:,0] >= 20) & (reimarray[:,0] <= 40))] = 1
            reimarray[:,0][np.where((reimarray[:,0] >= 40) & (reimarray[:,0] <= 75))] = 2
            reimarray[:,0][np.where((reimarray[:,0] >= 75) & (reimarray[:,0] <= 155))] = 3
            reimarray[:,0][np.where((reimarray[:,0] >= 155) & (reimarray[:,0] <= 190))] = 4
            reimarray[:,0][np.where((reimarray[:,0] >= 190) & (reimarray[:,0] <= 270))] = 5
            reimarray[:,0][np.where((reimarray[:,0] >= 270) & (reimarray[:,0] <= 295))] = 6
            reimarray[:,0][reimarray[:,0] > 295] = 7
            
            reimarray[:,1][reimarray[:,1] < 0.2] = 0
            reimarray[:,1][np.where((reimarray[:,1] >= 0.2) & (reimarray[:,1] <= 0.7))] = 1
            reimarray[:,1][reimarray[:,1] > 0.7] = 2
            
            reimarray[:,2][reimarray[:,2] < 0.2] = 0
            reimarray[:,2][np.where((reimarray[:,2] >= 0.2) & (reimarray[:,2] <= 0.7))] = 1
            reimarray[:,2][reimarray[:,2] > 0.7] = 2
        
            colorvector = reimarray[:,0] * 9 + reimarray[:,1] * 3 +reimarray[:,2]
            colorvector = np.histogram(colorvector,76)[0]
            return colorvector
        def get_colors(image_name):
            img = Image.open(image_name)
            color_feature = color_features(img)
            return color_feature
#        my_colors = []
#        for i in range(len(im_tags)):
#            my_colors.append(get_colors(self.im_list[im_tags[i]]))
#        my_colors = np.asarray(my_colors)
        features = self.features[im_tags]
#        features = features/np.max(features)
#        features = np.hstack((features,my_colors))
#        features = my_colors
        centers = features[np.isin(im_tags,self.sel_im)]
        #sel_im = sel_im.to_numpy()
        ward = KMeans(n_clusters=len(self.sel_im), init=centers, n_init=1).fit(features)
        test2 = ward.labels_
#        cluster_ind = []
#        for i in range(0,len(sel_im)):
#            cluster_ind.append([])
#        for k in range(0,len(test2)):
#            cluster_ind[test2[k]].append(k)
#        xlength = []
#        for q in range(0,len(cluster_ind)):
#            xlength.append(len(cluster_ind[q]))
#        cluster_indX = np.zeros((max(xlength),len(cluster_ind)))-1        
#        for r in range(0,len(cluster_ind)):
#            for s in range(0,len(cluster_ind[r])):
#                cluster_indX[s,r] = int(cluster_ind[r][s])
#        cluster_indX = cluster_indX.astype(int)
        


#        cur_cm = self.cm[im_tags]
#        cur_cm = cur_cm[:,sel_im]
#        cur_cm = np.nan_to_num(cur_cm)
#        cur_cmax = np.argmax(cur_cm,1)
        nco = np.array([],dtype='int')
        for kk in range(m):
            tb = im_tags[test2==kk]#.to_numpy()
            try:
                #nco= np.hstack([nco,sel_im[kk],self.currentcluster[cur_cmax==kk]])
                nco= np.hstack([nco,self.sel_im[kk],tb])

            except ValueError:
                #nco= np.vstack([nco,sel_im[kk],self.currentcluster[cur_cmax==kk]])
                nco= np.vstack([nco,self.sel_im[kk],tb])
        self.subcyes = 1
        self.display_images(nco)


    def purtangles(self):
        self.purpletangles = []
        self.sel_im
        self.purpletangles.append(self.c.create_rectangle(column_*(self.imsize + self.image_distance), row_*(self.imsize + self.image_distance), column_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, row_*(self.imsize + self.image_distance)+(self.imsize + self.image_distance)-self.image_distance, outline='cyan2',width=self.rectanglewidth,tags = self.ccluster[j]))
        
    def query_external(self):
        self.external_image = str(filedialog.askopenfilename())
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'query external image ' + self.external_image)
        self.externalfeatures = self.feature_extraction3(self.neuralnet,self.external_image)
        
        self.focuscm = self.create_matrix3(self.externalfeatures,self.features,'correlation')
        temp_list = np.sort(self.focuscm.transpose(),0)[::-1]
        
        self.focus_list = np.argsort(self.focuscm.transpose(),0)[::-1]
        #self.rank_list = self.rank_list[0:1000]
        self.c.xview_moveto(self.origX)  #####
        self.c.yview_moveto(self.origY) ######
        self.focus_list = np.asarray(self.focus_list)
        temp_list = np.asarray(temp_list)
        temp_list[np.isnan(temp_list)] = -100
        self.focus_list = self.focus_list[temp_list>-50]
        self.bucketDisp = 0
        self.c.delete("all")
        x = []
        for i in range(0,len(self.focus_list)):
            d = self.im_list[self.focus_list[i]]
            x.append(d)
        self.display_images(self.focus_list)

    
    def query_clipboard_image(self):
        # try:
            # Get image from clipboard
            clipboard_image = ImageGrab.grabclipboard()
            
            if clipboard_image is None:
                self.log.append(time.strftime("%H:%M:%S", time.gmtime()) + ' ' + 'No image in clipboard')
                self.communication_label.configure(text='No image in clipboard.')
                return
            
            if clipboard_image.mode == 'RGBA':
                clipboard_image = clipboard_image.convert('RGB')
            # Save the clipboard image to a BytesIO object
            image_bytes = io.BytesIO()
            clipboard_image.save(image_bytes, format='jpeg')
            image_bytes.seek(0)  # Move to the beginning of the BytesIO object
            
            self.external_image = image_bytes
            
            self.log.append(time.strftime("%H:%M:%S", time.gmtime()) + ' ' + 'Query clipboard image')
            
            self.externalfeatures = self.feature_extraction3(self.neuralnet, self.external_image)
            
            self.focuscm = self.create_matrix3(self.externalfeatures, self.features, 'correlation')
            temp_list = np.sort(self.focuscm.transpose(), 0)[::-1]
            
            self.focus_list = np.argsort(self.focuscm.transpose(), 0)[::-1]
            self.c.xview_moveto(self.origX)  #####
            self.c.yview_moveto(self.origY) ######
            self.focus_list = np.asarray(self.focus_list)
            temp_list = np.asarray(temp_list)
            temp_list[np.isnan(temp_list)] = -100
            self.focus_list = self.focus_list[temp_list > -50]
            self.bucketDisp = 0
            self.c.delete("all")
            x = []
            for i in range(0, len(self.focus_list)):
                d = self.im_list[self.focus_list[i]]
                x.append(d)
            self.display_images(self.focus_list)
            
        # except Exception as e:
        #     self.log.append(time.strftime("%H:%M:%S", time.gmtime()) + ' ' + 'Error querying clipboard image: ' + str(e))
    


    def expand_cluster(self):
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'expand cluster' + str(self.num_clus))
        cluster_e = self.df[:,self.num_clus]
        cluster_e = cluster_e[cluster_e > -1]
        current_features = np.mean(self.features[cluster_e],0)
        try:
            self.new_threshold = self.new_threshold
        except AttributeError:
                self.new_threshold = self.threshold
                
        def create_matrix3(current_features,features,distance_metric):
            current_features = np.squeeze(np.expand_dims(current_features,0))
            features = np.squeeze(np.expand_dims(features,0))
            features_t = np.transpose(features)
            current_features_t = np.transpose(current_features)
            cm = []                
            sumX = sum(features_t)
            focussumX = sum(current_features_t)
            sumsquareX = sum(features_t**2)
            
#                for i in range(0,features.shape[0]):
            feat0 = current_features
            sumXY = np.dot(feat0,features_t)
            r = features.shape[1]*sumXY - focussumX*sumX 
            s = ((features.shape[1] * sumsquareX) - sumX**2)
            t = 1./((s[0]*s)**0.5)
            u = r * t
            cm.append(u)
            cm = np.asarray(cm)
            return cm
        currentcm = np.nan_to_num(create_matrix3(current_features,self.features,'correlation').transpose())
        self.new_threshold = self.new_threshold - 0.1
        expanded_cluster = np.expand_dims(np.arange(0,len(self.im_list)),1)
        expanded_cluster = expanded_cluster[currentcm > self.new_threshold]
        expanded_cluster = np.setdiff1d(expanded_cluster,cluster_e)
        expanded_cluster = np.concatenate((expanded_cluster,cluster_e))
        self.display_images(expanded_cluster)

    # function to display the contents of the selected bucket 
    def showEdge(self):
#        with open('D:\\PhD\\Visual Analytics\\MockCase\\zForBucket.pickle', 'wb') as handle:
#            pickle.dump(self.theBuckets,handle,protocol=pickle.HIGHEST_PROTOCOL)        
        self.c.delete("all")
        self.oview = 0
        self.selected_edge = self.categories.get(self.categories.curselection()) #acquire the selected category
        self.current_edge = self.selected_edge
        if len(self.selected_edge) == 1:
            
            self.communication_label.configure(text='The hyperedge '+ self.categories.get(self.selected_edge) +' is shown.')


        self.display_hyperedges()

    def show_edge_from_side(self, edge_id):
        """Updates the selected edge and displays it."""
        self.c.delete("all")
        self.selected_edge = edge_id  # Set the selected edge directly
        self.current_edge = edge_id
        self.display_hyperedges()


    # function to add the currently displayed cluster to the selected bucket(s)
    def addCluster2bucket(self):
        self.refresh = 0
        selected = self.categories.curselection()
        bucket_com = ''
        
        for p in range(0, len(selected)):
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'add cluster' + str(self.num_clus) + ' to bucket(s) ' + str(self.categories.get(selected[p])))
            temp_c = self.theBuckets[self.categories.get(selected[p])]
            temp_c = np.asarray(temp_c)
            temp_x = np.concatenate((self.currentcluster,temp_c),axis = 0)
            temp_y = numpy_unique_ordered(temp_x)
            temp_y = np.asarray(temp_y).astype(int)
            self.theBuckets[self.categories.get(selected[p])] = temp_y
            bucket_com = bucket_com + str(self.categories.get(selected[p])) + ' '
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(temp_y) + ' added to ' + self.categories.get(selected[p]))
        self.communication_label.configure(text='cluster ' + str(self.num_clus) + ' has been added to ' + bucket_com)
        #self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+self.theBuckets)
        
    # function to add the currently selected image(s) to the selected bucket(s)
    def addSelection2bucket(self):
        self.refresh = 0
        selected = self.categories.curselection()
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'add selection to bucket ' + str(selected))
        bucket_com = ''
        for p in range(0, len(selected)):
            bucket_com = bucket_com + str(self.categories.get(selected[p])) + ' '
            temp_c = np.asarray(self.theBuckets[self.categories.get(selected[p])])
#            if self.bucketDisp==0:
            temp_z = self.ccluster[self.selected_images]
            temp_x = np.concatenate((temp_z,temp_c),axis = 0)
#            elif self.bucketDisp==1:
#                temp_z = self.ccluster[self.selected_images]
#                temp_x = np.concatenate((temp_z,temp_c),axis = 0)
#            elif self.bucketDisp==2:
#                temp_z = self.ccluster[self.selected_images]
#                temp_x = np.concatenate((temp_z,temp_c),axis = 0)
            temp_y = numpy_unique_ordered(temp_x)
            temp_y = np.asarray(temp_y).astype(int)            
            self.theBuckets[self.categories.get(selected[p])] = temp_y
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+str(temp_y) + ' added to ' + self.categories.get(selected[p]))
        #self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+self.theBuckets)
        if len(temp_z) == 1:
            self.communication_label.configure(text=str(len(temp_z)) + ' image has been added to ' + bucket_com)
        else:
            self.communication_label.configure(text=str(len(temp_z)) + ' images have been added to ' + bucket_com)
            
    # function to add a new user created bucket. Buckets are sorted alphabeteically.
    def addCategory(self):
        if self.e2.get() not in self.catList:
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'added new cat to list: ' + self.e2.get())
            self.categories.insert(END, self.e2.get())
            self.theBuckets[self.e2.get()] = []
            self.catList = self.categories.get(0,END)
            self.catList = sorted(self.catList, key=str.lower)
            boxscrollbar = Scrollbar(width = 10)
            #self.categories = Listbox(width=30,background='#777777',foreground='white',yscrollcommand=boxscrollbar.set,exportselection=0)
            #self.categories['selectmode'] = 'extended'               
            #self.categories.place(x=585,y=50)
            self.categories.delete(0,END)
            for k in range(0,len(self.catList)):
                self.categories.insert(END,self.catList[k])
            boxscrollbar.config(command=self.categories.yview)
            boxscrollbar.place(in_=self.categories,relx=1.0, relheight=1)
            self.categories.select_set(self.catList.index(self.e2.get()))
            self.categories.see(self.catList.index(self.e2.get()))
            self.categories.bind('<Button-1>', self.deselect_list )
    #function to display the next bucket
    def nextCluster(self):
        if self.num_clus < self.df.shape[1]-1:
            self.num_clus += 1
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'clicked next cluster: '+ str(self.num_clus))
            #self.b1['text'] ="next cluster" + ' (' + str(self.num_clus) + ')'
            self.c.xview_moveto(self.origX)
            self.c.yview_moveto(self.origY)
            self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters')
            self.new_threshold = self.threshold
            
            
            
    
    #function to display the previous bucket
    def prevCluster(self):
        if self.num_clus > 0:
            self.num_clus -= 1
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'clicked previous cluster: '+ str(self.num_clus))
            self.c.xview_moveto(self.origX)
            self.c.yview_moveto(self.origY)
            self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters')
            self.new_threshold = self.threshold

    def showCluster(self):
        self.num_clus = int(self.ecluster.get())-1
        self.c.xview_moveto(self.origX)
        self.c.yview_moveto(self.origY)
        self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters')
        self.new_threshold = self.threshold
        

    def change_name(self):
        new_bucket = self.e2.get()
        old_bucket = self.categories.curselection()
        if len(old_bucket) == 1:
            old_bucket = self.catList[old_bucket[0]]
            self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'changed name of bucket'+ old_bucket + ' into '+ new_bucket)
            self.catList = list(self.catList)
            self.catList[self.catList.index(old_bucket)] = new_bucket
            self.theBuckets[new_bucket] = self.theBuckets.pop(old_bucket)
            self.catList = sorted(self.catList, key=str.lower)
            boxscrollbar = Scrollbar(width = 10)
            self.categories.delete(0,END)
            for k in range(0,len(self.catList)):
                self.categories.insert(END,self.catList[k])
            boxscrollbar.config(command=self.categories.yview)
            boxscrollbar.place(in_=self.categories,relx=1.0, relheight=1)
            self.categories.select_set(self.catList.index(self.e2.get()))
            self.categories.see(self.catList.index(self.e2.get()))
            self.categories.bind('<Button-1>', self.deselect_list )
            
    #function to display the images of the current cluster, e.g. to switch from the bucket view back to the current cluster.
    def showImg(self):
        self.bucketDisp = 0
        self.imagex = []
        self.c.delete("all")
        # self.num_clus
        # self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'showing cluster ' + str(self.num_clus))        
        
        try:
            cluster = np.asarray(self.df[:,self.num_clus])
            cluster = cluster[cluster > -1]
            self.currentcluster = cluster
            self.greentangles = []
            self.communication_label.configure(text='You are currently viewing cluster ' + str(self.num_clus+1) +' out of ' + str(self.df.shape[1]) + ' clusters. This cluster contains ' + str(len(cluster)) + ' images.')
        except KeyError:
            cluster = []
        self.display_images(cluster)
    #function to save the current session as a pickle file. The buckets, the list of images, the clusters, the current cluster number, the array with correlations, and the array with features are saved.
    

    def initialize_hyperedges(self):
        n_images, n_hyperedges = self.df.shape
        self.hyperedges = OrderedDict()
        self.image_mapping = {}
        for h_idx in range(n_hyperedges):
            hyperedge_name = f'edge_{h_idx}'
            print('h_idx',h_idx)
            # Get the column corresponding to the hyperedge
            column = self.df[:, h_idx]
            # Find image indices where the entry is 1
            image_indices = np.nonzero(column)[0]
            # Update hyperedges dictionary
            self.hyperedges[hyperedge_name] = set(image_indices)
            
            # Update images dictionary
            for i_idx in image_indices:
                if i_idx not in self.image_mapping:
                    self.image_mapping[i_idx] = set()
                self.image_mapping[i_idx].add(hyperedge_name)
        self.filtered_hyperedges = self.hyperedges
        


    def save_as(self):
        self.catList = self.categories.get(0, END)
        self.answer = filedialog.asksaveasfilename(defaultextension=".h5")
        
        with h5py.File(self.answer, 'a') as hdf:
            def create_or_replace_dataset(name, savedata):
                if name in hdf:
                    del hdf[name]
                hdf.create_dataset(name, data=savedata)
            
            h5_im_list = [n.encode("utf-8", "ignore") for n in self.im_list]
            create_or_replace_dataset('im_list', h5_im_list)
    
            # h5_log = [n.encode("utf-8", "ignore") for n in self.log]
            # create_or_replace_dataset('log', h5_log)
    
            create_or_replace_dataset('df', self.df)
    
            h5_catlist = [n.encode("utf-8", "ignore") for n in list(self.catList)]
            create_or_replace_dataset('catList', h5_catlist)

            # Save hyperedges dictionary           
            hyperedge_names = list(self.hyperedges.keys())
            hyperedge_image_ids = [np.array(list(image_ids), dtype='int64') for image_ids in self.hyperedges.values()]
            dt = h5py.vlen_dtype(np.dtype('int64'))
            create_or_replace_dataset('hyperedge_names', np.array(hyperedge_names, dtype=h5py.string_dtype()))
            create_or_replace_dataset('hyperedge_image_ids', np.array(hyperedge_image_ids, dtype=dt))
            # Save image_mapping dictionary
            image_ids = list(self.image_mapping.keys())
            image_hyperedge_names = [np.array(list(hyperedge_names), dtype=h5py.string_dtype()) for hyperedge_names in self.image_mapping.values()]
            dt_str = h5py.vlen_dtype(h5py.string_dtype())
            create_or_replace_dataset('image_ids', np.array(image_ids, dtype='int64'))
            create_or_replace_dataset('image_hyperedge_names', np.array(image_hyperedge_names, dtype=dt_str))

            create_or_replace_dataset('num_clus', self.num_clus)
    
            path_list = [self.hdf_path]
            p_list = [n.encode("utf-8", "ignore") for n in path_list]
            create_or_replace_dataset('cm_path', p_list)
    
            create_or_replace_dataset('features', self.features)
            # create_or_replace_dataset('X_embed', self.x_test)
    
            self.communication_label.configure(text='Saved!')


    def create_correlation_vector(self, focusfeatures, features):
        """
        Compute the Pearson correlation coefficient between a focus feature vector and each sample in the features matrix.

        Parameters:
            focusfeatures (array-like): 1D array of shape (d,)
            features (array-like): 2D array that is either of shape (d, n) or (n, d). 
                                If it’s (n, d) (each sample is a row), it will be transposed.
        Returns:
            np.ndarray: 1D array of correlation coefficients for each sample.
        """
        # Ensure focusfeatures is 1D and features is 2D.
        focusfeatures = np.ravel(focusfeatures)
        features = np.atleast_2d(features)
        
        # If features is (n, d) where d == len(focusfeatures), transpose it to (d, n)
        if features.shape[0] != focusfeatures.shape[0] and features.shape[1] == focusfeatures.shape[0]:
            features = features.T
        elif features.shape[0] != focusfeatures.shape[0] and features.shape[1] != focusfeatures.shape[0]:
            raise ValueError("The dimensions of 'features' do not match the length of 'focusfeatures'.")
        
        # Number of samples (each column is one sample)
        n = features.shape[1]
        
        # Compute necessary sums
        sum_focus = np.sum(focusfeatures)               # scalar
        sum_features = np.sum(features, axis=0)           # shape: (n,)
        sum_focus_sq = np.sum(focusfeatures**2)           # scalar
        sum_features_sq = np.sum(features**2, axis=0)       # shape: (n,)
        
        # Dot product: yields a vector of shape (n,)
        dot_product = np.dot(focusfeatures, features)
        
        # Compute numerator and denominator for Pearson correlation
        numerator = n * dot_product - sum_focus * sum_features
        denominator = np.sqrt((n * sum_focus_sq - sum_focus**2) *
                            (n * sum_features_sq - sum_features**2))
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation = np.where(denominator != 0, numerator / denominator, 0)
        
        return correlation

    def rank_images(self):
        self.oview = 0
        self.focuscm = self.create_correlation_vector(self.focusfeatures, self.features)
        sorted_indices = np.argsort(self.focuscm)[::-1]

        self.focuscm = np.nan_to_num(self.focuscm, nan=-100)
        filtered_indices = sorted_indices[self.focuscm[sorted_indices] > -50]
        self.focus_list = filtered_indices        
        # images_to_display = [self.im_list[i] for i in self.focus_list]
        self.display_images(self.focus_list)
        if len(self.focus_list) > 1000:
            self.focus_list = self.focus_list[0:1000]

        data = {}
        for image in self.focus_list:
            row_dict = {}   
            row_dict['label'] = str(image)  # display image as row label

            for edge in self.all_edges:
                row_dict[edge] = '1' if image in self.hyperedges[edge] else '0'
            
            data[image] = row_dict

        # Create a model from this data
        model = TableModel()
        model.importDict(data)
        mview = SyncScrollExample(self.tab1, self.focus_list, self.hdf_path, data, model)  # Try bigger row_count
        mview.grid(row=0, column=0, sticky="nsew")

        self.tab1.grid_columnconfigure(0, weight=1)
        self.tab1.grid_rowconfigure(0, weight=1)


        self.tab1.bind('<Button-3>', self.click_select_in_matrix)
        if len(self.previous_visible_points) > 0:
            self.redraw_hypergraph(update=None,ranking=True)





    def get_img_idx_from_event(self, event):
            if not isinstance(event.widget, Canvas):
                return

            canvas = event.widget
            items = canvas.find_overlapping(event.x, event.y, event.x, event.y)
            
            for item in reversed(items):
                if canvas.type(item) == 'image':
                    tags = canvas.gettags(item)
                    if tags:
                        self.img_idx = int(tags[0])  
                        print("Clicked image index:", self.img_idx)
                        self.focusfeatures = self.features[self.img_idx]
                        self.rank_images()
            return None
            
    #function to load a previous session. 
    def load_as(self):
        self.answer2 = filedialog.askopenfilename(defaultextension=".h5")  
        self.bind_all("<Shift-Button-3>", self.get_img_idx_from_event)
        with h5py.File(self.answer2, 'r') as hdf:
            # Load image list
            self.im_list = hdf['file_list'][()]
            if isinstance(self.im_list[0], bytes):
                self.im_list = [n.decode("utf-8", "ignore") for n in self.im_list]

            self.df = hdf['clustering_results'][()]

            if 'catList' in hdf:
                self.catList = list(hdf['catList'][()])
                if isinstance(self.catList[0], bytes):
                    self.catList = [n.decode("utf-8", "ignore") for n in self.catList]
            else:
                num_columns = self.df.shape[1]
                self.catList = tuple(f'edge_{i}' for i in range(num_columns))

            if 'hyperedges' in hdf:
                hyperedge_names = hdf['hyperedge_names'][()]
                hyperedge_image_ids = hdf['hyperedge_image_ids'][()]
                self.hyperedges = OrderedDict(
                    (name, set(ids)) for name, ids in zip(hyperedge_names, hyperedge_image_ids)
                )

                image_ids = hdf['image_ids'][()]
                image_hyperedge_names = hdf['image_hyperedge_names'][()]
                self.image_mapping = {
                    id: set(names) for id, names in zip(image_ids, image_hyperedge_names)
                }
            else:
                self.initialize_hyperedges()

            if 'num_clus' in hdf:
                self.num_clus = hdf['num_clus'][()]
            else:
                self.num_clus = 0  

            self.hdf_path = self.answer2
            self.features = hdf['features'][()]

        self.categories.delete(0, END)
        for category in self.catList:
            self.categories.insert(END, category)

        boxscrollbar = Scrollbar(width=10)
        boxscrollbar.config(command=self.categories.yview)
        boxscrollbar.place(in_=self.categories, relx=1.0, relheight=1)
        self.categories.bind('<Button-1>', self.deselect_list)


    #function to export (make a copy of) all the images in all the buckets to folders with the buckets' names, in a location specified by the user.        
    def export_buckets(self):
        self.catList = self.categories.get(0,END)        
        self.answer = filedialog.askdirectory()   #this will make the file directory a string
        for p in range(0,len(self.catList)):
            bucket = self.theBuckets[self.catList[p]]
            if len(bucket) > 0:
                bucket_dir = self.answer + '/' + self.catList[p]
                if not os.path.exists(bucket_dir):
                        os.makedirs(bucket_dir)
                for q in range(0,len(bucket)):
                    source_ = self.im_list[int(bucket[q])]
                    destination_ = bucket_dir + "/" + os.path.basename(source_)
                    if os.path.isfile(destination_):
                        shutil.copy2(source_, destination_ + id_generator())
                    else:
                        shutil.copy2(source_, destination_)
            
    # #function to select the image folder that the user wants to analyze
    # def select_folder(self):
    #     self.selected_folder = filedialog.askdirectory()
    #     if self.selected_folder:
    #         self.im_list = get_image_files(self.selected_folder)

    def _clear_temp_data(self):
        self.temp_image_paths = None
        self.temp_features = None
        self.temp_thumbnail_data = None
        # self.temp_thumbnails_are_embedded will be reset upon new extraction
        if self.save_project_button:
             self.save_project_button.config(state=DISABLED)


    def run_extraction(self):
        if self.temp_features is not None and self.temp_image_paths is not None:
            use_cached = messagebox.askyesno(
                "Cached Data Found",
                "Previously extracted features are available.\n"
                "Do you want to use this data and proceed to save?\n\n"
                "(Choosing 'No' will re-extract features from a new folder)",
                parent=self.master
            )
            if use_cached:
                print("Using cached feature data. Prompting to save project.")
                self.save_project_button.config(state=NORMAL)
                self.prompt_and_save_project() # This will handle thumbnail generation
                return
            else:
                self._clear_temp_data() # Clear old temp features

        # --- Proceed with new feature extraction ---
        self.extract_button.config(state=DISABLED)
        self.save_project_button.config(state=DISABLED)
        self.master.update_idletasks()

        selected_folder = filedialog.askdirectory(title="Select Folder Containing Images", parent=self.master)
        if not selected_folder: # ... (handle cancellation) ...
            self.extract_button.config(state=NORMAL); return

        try:
            image_paths = get_image_files(directory=selected_folder) # ... (handle no images) ...
            if not image_paths:
                messagebox.showwarning("No Images", f"No images found in '{selected_folder}'.", parent=self.master)
                self._clear_temp_data()
                self.extract_button.config(state=NORMAL)
                return

            print(f"Found {len(image_paths)} images. Extracting features...")
            extractor = FeatureExtractor(model_name=self.neuralnet, batch_size=16)
            features = extractor.extract_features(file_list=image_paths)
            
            self.temp_image_paths = image_paths
            self.temp_features = features
            # Thumbnails are NOT generated here. They are generated during the save process.

            messagebox.showinfo("Extraction Complete",
                                f"Features for {len(self.temp_image_paths)} images extracted.\n"
                                "Click 'Save Project' to generate thumbnails and save.",
                                parent=self.master)
            self.save_project_button.config(state=NORMAL)

        except Exception as e: # ... (error handling) ...
            print(f"Error during extraction: {e}"); import traceback; traceback.print_exc()
            messagebox.showerror("Extraction Error", f"An error occurred: {e}", parent=self.master)
            self._clear_temp_data()
        finally:
            self.extract_button.config(state=NORMAL)

        def prompt_and_save_project(self):
            if self.temp_features is None or self.temp_image_paths is None:
                messagebox.showwarning("No Data", "No features extracted yet. Please extract features first.", parent=self.master)
                return

            # Potentially add a Checkbutton to the UI to set self.store_thumbnails_externally_setting
            # For now, it's a class attribute.
            embed_thumbnails_in_hdf5_for_this_save = not self.store_thumbnails_externally_setting

            hdf5_file_path = filedialog.asksaveasfilename( # ... (as before) ...
                title="Save Project As",
                defaultextension=".h5",
                initialfile="my_image_project.h5",
                filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
                parent=self.master
            )
            if not hdf5_file_path: # ... (handle cancellation) ...
                return

            self.save_project_button.config(state=DISABLED)
            self.master.update_idletasks()

            try:
                print(f"Preparing to save project. Thumbnails will be embedded: {embed_thumbnails_in_hdf5_for_this_save}")
                
                external_thumb_dir_abs = None
                if not embed_thumbnails_in_hdf5_for_this_save: # i.e., store_externally is True
                    project_dir = os.path.dirname(hdf5_file_path)
                    external_thumb_dir_abs = os.path.join(project_dir, "thumbnails")
                    print(f"External thumbnail directory: {external_thumb_dir_abs}")

                # Generate thumbnail data (bytes or paths) based on the setting
                # This now happens *during* the save operation, after HDF5 path is known.
                thumbnail_data_for_saving = self.hdf5_handler.generate_thumbnails_data(
                    image_file_paths=self.temp_image_paths,
                    thumbnail_size=(128, 128),
                    store_externally=(not embed_thumbnails_in_hdf5_for_this_save),
                    external_thumbnail_dir=external_thumb_dir_abs # Will be None if embedding
                )

                clustering_results_placeholder = np.array([], dtype='i8')

                self.hdf5_handler.save_project_to_hdf5(
                    hdf5_file_path=hdf5_file_path,
                    image_file_paths=self.temp_image_paths,
                    features=self.temp_features,
                    thumbnail_data=thumbnail_data_for_saving, # This is now bytes or paths
                    thumbnails_are_embedded=embed_thumbnails_in_hdf5_for_this_save,
                    clustering_results=clustering_results_placeholder
                )

                # Update active project state
                self.project_file_path = hdf5_file_path
                self.im_list = list(self.temp_image_paths)
                self.features = np.array(self.temp_features)
                self.app_thumbnail_data = thumbnail_data_for_saving # Store the actual data (bytes/paths)
                self.app_thumbnails_are_embedded = embed_thumbnails_in_hdf5_for_this_save

                messagebox.showinfo("Project Saved", f"Project successfully saved to:\n{hdf5_file_path}", parent=self.master)
                self._clear_temp_data()
                self.load_ui_with_project_data()

            except Exception as e: # ... (error handling) ...
                print(f"Error during project save: {e}"); import traceback; traceback.print_exc()
                messagebox.showerror("Save Error", f"Could not save project: {e}", parent=self.master)
                # self.save_project_button.config(state=NORMAL) # Allow retry if temp data still exists
            finally:
                if self.temp_features is not None: # If save failed but temp data is there
                    self.save_project_button.config(state=NORMAL)
                # else button remains disabled as temp data was cleared on success



    def select_folder_and_extract_features(self,
        model_name: str,
        batch_size: int = 32,        
    )-> Tuple[Optional[np.ndarray], Optional[List[str]]]: 
        """
        Prompts the user to select a folder, scans for images using get_image_files,
        and then extracts features using FeatureExtractor.

        Args:
            model_name (str): The name of the timm model to use for feature extraction.
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
            parent_window (Optional[tk.Tk], optional): An existing Tkinter root window.
                                                    If None, a temporary one is created.
                                                    Defaults to None.

        Returns:
            Tuple[Optional[np.ndarray], Optional[List[str]]]:
                A tuple containing:
                - Extracted features as a NumPy array (or None if an error occurs or cancelled).
                - List of image file paths used for extraction (or None).
        """
        folder_path = filedialog.askdirectory(
            title="Select Folder Containing Images",
        )


        if not folder_path:
            print("No folder selected by the user.")
            return None, None

        print(f"Selected folder: {folder_path}")

        try:
            print("Scanning for image files...")
            image_paths = get_image_files(directory=folder_path)

            if not image_paths:
                print(f"No images found in '{folder_path}' or its subdirectories.")
                return None, None

            print(f"Found {len(image_paths)} images.")
            # You might want to show a progress bar or message in a real Tkinter app here

            print(f"Initializing feature extractor with model: {model_name}...")
            # Consider num_workers=0 for DataLoader if issues arise, especially on Windows
            # or when not using if __name__ == '__main__': for the main script execution.
            extractor = FeatureExtractor(model_name=model_name, batch_size=batch_size)

            print("Extracting features... This may take a while.")
            features = extractor.extract_features(file_list=image_paths)
            print(f"Successfully extracted features. Shape: {features.shape}")

            return features, image_paths

        except Exception as e:
            error_msg = f"An error occurred: {e}"
            print(error_msg)
            return None, None

    #function to extract features from a neural network as defined below. New networks can be added if needed.
    def feature_extraction(self, neural_net, im_list,new_imgs=False):
            f = im_list
            with h5py.File(self.hdf_path, 'a') as hdf:
                try:
                    hdf.create_dataset('thumbnail_images',((len(im_list),100,100,3)),chunks=(1,100,100,3),maxshape=(None,100,100,3))
                except RuntimeError:
                    pass

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if neural_net == 'inception_v3':
                f_size = 2048
                model = models.inception_v3(pretrained='imagenet')
                layer = model._modules.get('Mixed_7c')
            elif neural_net == 'resnet152': #2084
                f_size = 2048
                model = models.resnet152(pretrained=True)
                layer = model._modules.get('avgpool')
            elif neural_net == 'resnet18': #512
                f_size =512
                model = models.resnet18(pretrained=True)
                layer = model._modules.get('avgpool')            
            elif neural_net == 'swin2': #512
                f_size = 1000
                model = timm.create_model('swinv2_large_window12to16_192to256_22kft1k',pretrained=True)
                layer = 'nothing'
            elif neural_net == 'vgg16': #4096
                f_size = 4096
                model = models.vgg16(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer   = 'nothing'
            elif neural_net == 'vgg19': #4096
                f_size =    4096
                model = models.vgg19(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            elif neural_net == 'densenet161': #2208
                f_size =2208
                model = models.densenet161(pretrained=True)	
                model = model.features
            elif neural_net == 'squeezenet1_0': #512
                f_size = 1000
                model = models.squeezenet1_0(pretrained=True)
                #model = model.features
                layer = 'nothing'
            elif neural_net == 'alexnet':    
                f_size = 4096
                model = models.alexnet(pretrained=True)
                model.classifier = model.classifier[:-1]
                layer = 'nothing'
            elif neural_net == 'clip':
                f_size = 1024
                #model, preprocess = clip.load("ViT-B/32", device=device)
                model, preprocess = clip.load("RN50x64", device=device)
                layer = 'nothing'
                
            model.eval()
            model = model.to(device)
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            if neural_net == 'inception_v3':
                transform = transforms.Compose([
                            transforms.Resize((299,299)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            elif neural_net == 'swin2':
                transform = transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                    ])
            else:
                transform = transforms.Compose([
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
            def get_vector(image_name,f_size, layer, transform, model, neural_net):
                try:
                    image_name = image_name.strip()
                    img = Image.open(image_name)
                        
                        
                except OSError:
                    #if an image fails to open, a black replacement image is created...
                    print('failopen',image_name)
                    img = Image.new('RGB',(100,100))
                
                if neural_net == 'clip':
                    pass
                else:
                    if img.mode == 'RGB':
                        try:
                            t_img = transform(img).unsqueeze(0)
                        except OSError:
                            t_img = transform(img).unsqueeze(0)
                        
                        
                        t_img = t_img.to(device)
                        if neural_net == 'alexnet' or neural_net =='vgg19' or neural_net =='vgg16' or neural_net=='swin2' or neural_net =='alexnet' or neural_net =='squeezenet1_0':
                            torch.cuda.empty_cache()
                            my_embeddingz = model(t_img)
                            my_embeddingz = my_embeddingz.cpu()
                            my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                        elif neural_net == 'densenet161':
                            featuresY = model(t_img)
                            my_embeddingz = F.relu(featuresY,inplace= True)
                            my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=7, stride=1).view(featuresY.size(0), -1)
                            my_embeddingz = my_embeddingz.cpu()
                            my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                        elif neural_net == 'inception_v3':
                            my_embeddingz = torch.zeros((1,f_size,8,8))
                            my_embeddingz = my_embeddingz.to(device)
                                # 4. Define a function that will copy the output of a layer
                            def copy_data(m, i, o):
                                my_embeddingz.copy_(o.data)
                                # 5. Attach that function to our selected layer
                            h = layer.register_forward_hook(copy_data)
                                # 6. Run the model on our transformed image
                            model(t_img)
                            #    # 7. Detach our copy function from the layer
                            h.remove()
                            my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=8)
                            my_embeddingz = my_embeddingz.view(my_embeddingz.size(0), -1)
                            my_embeddingz = my_embeddingz.cpu()
                            my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                        elif neural_net == 'clip':
                            t_img = preprocess(img).unsqueeze(0).to(device)
                            with torch.no_grad():
                                my_embeddingz = model.encode_image(t_img)
                                my_embeddingz = my_embeddingz.cpu()
                                my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())                                                            
                        else:
                            my_embeddingz = torch.zeros((1,f_size,1,1))
                            my_embeddingz = my_embeddingz.to(device)
                                # 4. Define a function that will copy the output of a layer
                            def copy_data(m, i, o):
                                my_embeddingz.copy_(o.data)
                                # 5. Attach that function to our selected layer
                            h = layer.register_forward_hook(copy_data)
                                # 6. Run the model on our transformed image
                            model(t_img)
                            #    # 7. Detach our copy function from the layer
                            h.remove()
                            my_embeddingz = my_embeddingz.cpu()
                            my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                    else:
                        my_embeddingz = np.zeros((f_size,))
                return my_embeddingz,img
            my_embed = []
#            self.progress["value"] = 0
#            self.progress["maximum"] = len(f)-1
            with h5py.File(self.hdf_path, 'a') as hdf:            
                for i in range(0,len(f)):
                    veccie,img_to_store = get_vector(f[i],f_size,layer,transform,model,neural_net)
                    my_embed.append(veccie)
                    img_to_store = np.array(img_to_store.resize((100,100)))

                    if len(img_to_store.shape) == 2:
                        img_to_store = np.stack((img_to_store,)*3, axis=-1)
                    elif img_to_store.shape[2] == 4:
                        img_to_store = img_to_store[:,:,0:3]
                    elif img_to_store.shape[2] == 2:
                        img_to_storex = np.zeros((100,100,3))+200
                        img_to_storex[:,:,0:2] = img_to_store
                        img_to_store = img_to_storex
                    if new_imgs == True:
                        hdf['thumbnail_images'][i+len(self.features)] = img_to_store
                    else:
                        hdf['thumbnail_images'][i] = img_to_store
    #                if i%10 == 0:
    #                self.communication_label.update()
            print("Finished extracting")
            features = np.asarray(my_embed)
            
            return features

    def feature_extraction3(self,neural_net, image_source):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if neural_net == 'inception_v3':
            f_size = 2048
            model = models.inception_v3(pretrained='imagenet')
            layer = model._modules.get('Mixed_7c')
        elif neural_net == 'resnet152': #2084
            f_size = 2048
            model = models.resnet152(pretrained=True)
            layer = model._modules.get('avgpool')
        elif neural_net == 'resnet18': #512
            f_size =512
            model = models.resnet18(pretrained=True)
            layer = model._modules.get('avgpool')
        elif neural_net == 'vgg16': #4096
            f_size = 4096
            model = models.vgg16(pretrained=True)
            model.classifier = model.classifier[:-1]
            layer   = 'nothing'
        elif neural_net == 'vgg19': #4096
            f_size =    4096
            model = models.vgg19(pretrained=True)
            model.classifier = model.classifier[:-1]
            layer = 'nothing'
        elif neural_net == 'densenet161': #2208
            f_size =2208
            model = models.densenet161(pretrained=True)    
            model = model.features
        elif neural_net == 'squeezenet1_0': #512
            f_size = 1000
            model = models.squeezenet1_0(pretrained=True)
            #model = model.features
            layer = 'nothing'
        elif neural_net == 'alexnet':    
            f_size = 4096
            model = models.alexnet(pretrained=True)
            model.classifier = model.classifier[:-1]
            layer = 'nothing'
        model.eval()
        model = model.to(device)
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        if neural_net == 'inception_v3':
            transform = transforms.Compose([
                        transforms.Resize((299,299)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
        else:
            transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
        def get_vector3(image_source, f_size, layer, transform, model, neural_net):
            img = Image.open(image_source)
            
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation] == 'Orientation':
                        break
                exif = dict(img._getexif().items())
            
                if exif[orientation] == 3:
                    img = img.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    img = img.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    img = img.rotate(90, expand=True)
            except AttributeError:
                pass
            except KeyError:
                pass
    
            if img.mode == 'RGB':
                try:
                    t_img = transform(img).unsqueeze(0)
                except OSError:
                    t_img = transform(img).unsqueeze(0)
                t_img = t_img.to(device)
                if neural_net in ['alexnet', 'vgg19', 'vgg16', 'squeezenet1_0']:
                    torch.cuda.empty_cache()
                    my_embeddingz = model(t_img)
                    my_embeddingz = my_embeddingz.cpu()
                    my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                elif neural_net == 'densenet161':
                    featuresY = model(t_img)
                    my_embeddingz = F.relu(featuresY, inplace=True)
                    my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=7, stride=1).view(featuresY.size(0), -1)
                    my_embeddingz = my_embeddingz.cpu()
                    my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                elif neural_net == 'inception_v3':
                    my_embeddingz = torch.zeros((1, f_size, 8, 8))
                    my_embeddingz = my_embeddingz.to(device)
                    def copy_data(m, i, o):
                        my_embeddingz.copy_(o.data)
                    h = layer.register_forward_hook(copy_data)
                    model(t_img)
                    h.remove()
                    my_embeddingz = F.avg_pool2d(my_embeddingz, kernel_size=8)
                    my_embeddingz = my_embeddingz.view(my_embeddingz.size(0), -1)
                    my_embeddingz = my_embeddingz.cpu()
                    my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
                else:
                    my_embeddingz = torch.zeros((1, f_size, 1, 1))
                    my_embeddingz = my_embeddingz.to(device)
                    def copy_data(m, i, o):
                        my_embeddingz.copy_(o.data)
                    h = layer.register_forward_hook(copy_data)
                    model(t_img)
                    h.remove()
                    my_embeddingz = my_embeddingz.cpu()
                    my_embeddingz = np.ndarray.flatten(my_embeddingz.data.numpy())
            else:
                my_embeddingz = np.zeros((f_size,))
            return my_embeddingz
        my_embed = []
        self.progress["value"] = 0
        self.progress["maximum"] = 0  # Update this with the actual number of images if you have a progress bar.
        
        my_embed.append(get_vector3(image_source, f_size, layer, transform, model, neural_net))
    
        features = np.asarray(my_embed)
        return features

    
    def create_matrix(self,features,distance_metric):
            cm = []
            if distance_metric == 'correlation':
                features = np.squeeze(np.expand_dims(features,0))
                features_t = np.transpose(features)
                cm = []
                sumX = sum(features_t)
                sumsquareX = sum(features_t**2)
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    sumXY = np.dot(feat0,features_t)
                    r = features.shape[1]*sumXY - sumX[i]*sumX 
                    s = ((features.shape[1] * sumsquareX) - sumX**2)
                    t = 1./((s[i]*s)**0.5)
                    u = r * t
                    cm.append(u)
                cm = np.asarray(cm)
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
            elif distance_metric == 'l2':
                cm= []
                features = np.squeeze(np.expand_dims(features,0))
                #features = preprocessing.normalize(features, norm='l2')
                features_t = np.transpose(features)
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    featuresX = features - feat0
                    featuresX = featuresX**2
                    sumIT = np.sum(featuresX,1)
                    sumIT = sumIT ** 0.5
                    cm.append(sumIT)
                cm = np.asarray(cm)
                cm = 1-cm
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
        
            elif distance_metric == 'l1':
                cm= []
                features = np.squeeze(np.expand_dims(features,0))
                features_t = np.transpose(features)
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    featuresX = features - feat0
                    featuresX = featuresX**2
                    sumIT = np.sum(featuresX,1)
                    cm.append(sumIT)
                cm = np.asarray(cm)
                cm = 1-cm            
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
            elif distance_metric == 'euclidean':
                cm= []
                features = np.squeeze(np.expand_dims(features,0))
                features_t = np.transpose(features)
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    featuresX = features - feat0
                    featuresX = featuresX**2
                    sumIT = np.sum(featuresX,1)
                    sumIT = sumIT / np.max(sumIT)
                    cm.append(sumIT)
                   
                cm = np.asarray(cm)
                cm = 1-cm            
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
        
            elif distance_metric == 'cosine':
                cm = []
                features = np.squeeze(np.expand_dims(features,0))
                features_t = np.transpose(features)
                rootsumsquareX = sum(features_t**2)**0.5
                for i in range(0,features.shape[0]):
                    feat0 = features[i]
                    sXY = np.dot(feat0,features_t).transpose()
                    t_fea = sXY/(rootsumsquareX*rootsumsquareX[i])
                    cm.append(t_fea)
                cm = np.asarray(cm).astype('float32')
                for i in range(0,features.shape[0]):
                    cm[i,i]=None
        
            return cm


    def create_matrix3(self, focusfeatures, features, distance_metric):
        # Ensure inputs are at least 2D; this avoids accidental squeezing
        focusfeatures = np.atleast_2d(focusfeatures)
        features = np.atleast_2d(features)
        
        # Number of elements in each vector
        N = features.shape[1]
        
        # Compute sums
        sum_x = np.sum(focusfeatures)
        sum_y = np.sum(features)
        
        # Compute dot product and sums of squares
        sum_xy = np.dot(focusfeatures, features.T)  # Should be a 1x1 array
        sum_x2 = np.sum(focusfeatures**2)
        sum_y2 = np.sum(features**2)
        
        # Compute numerator of Pearson's correlation coefficient
        numerator = N * sum_xy - sum_x * sum_y
        
        # Compute denominator for each vector
        denom_x = N * sum_x2 - sum_x**2
        denom_y = N * sum_y2 - sum_y**2
        
        # Avoid division by zero
        if denom_x == 0 or denom_y == 0:
            return np.array([[0]])
        
        # Compute correlation coefficient
        corr = numerator / np.sqrt(denom_x * denom_y)
        return corr


    
    def calculate_features(self):
        self.hdf_path = filedialog.asksaveasfilename(defaultextension=".h5")

        self.features = self.feature_extraction(self.neuralnet,self.im_list)
        self.communication_label.configure(text = 'Calculated the features. You can now start clustering the images by pressing Cluster images. You may also want to save now that the features are calculated')
        self.communication_label['background'] = '#FFFFFF'

        def create_matrix_fast(features,hdf_path):            
            block_size = 1000
            len_f = len(features)
        
            with h5py.File(hdf_path, 'a') as hdf:
                try:
                    hdf.create_dataset('cm',((len_f,len_f)),chunks=(1,len_f),maxshape=(None,None))#(,len_f)
                except RuntimeError:
                    del hdf['cm']
                    hdf.create_dataset('cm',((len_f,len_f)),chunks=(1,len_f),maxshape=(None,None))#(len_f,len_f)
        
                loops = math.floor(len_f/block_size)
                start_loop1 = -1
                for loop2 in range(loops):        
                    features_p2 = features[loop2*block_size:loop2*block_size + block_size]
                    start_loop1 = start_loop1 + 1
                    for loop1 in range(start_loop1, loops):
                        features_p1 = features[loop1*block_size:loop1*block_size + block_size]
                        cm_p = np.corrcoef(features_p1,features_p2)[block_size::,0:block_size]
                        if loop1 == loop2:
                            for id_1 in range(len(features_p1)):
                                cm_p[id_1,id_1]=0
        
                        hdf['cm'][loop2*block_size:loop2*block_size + block_size, loop1*block_size:loop1*block_size + block_size] = cm_p
        
                        if loop1 != loop2:
                            hdf['cm'][loop1*block_size:loop1*block_size + block_size,loop2*block_size:loop2*block_size + block_size] = cm_p.transpose()
        
                remaining = len_f - block_size * loops
                if remaining > 0:
                    features_p1 = features[block_size*loops::]
                    for loop2 in range(loops):
                        features_p2 = features[loop2*block_size:loop2*block_size + block_size]
                        cm_p = np.corrcoef(features_p2,features_p1)[0:block_size,block_size::]
                        hdf['cm'][loop2*block_size:loop2*block_size + block_size, block_size*loops::] = cm_p
        
                    cm_p = np.corrcoef(features_p1)
                    for id_1 in range(len(cm_p)):
                        cm_p[id_1,id_1]=0
        
                    hdf['cm'][block_size*loops::,block_size*loops::] = cm_p
                    hdf['cm'][block_size*loops::,:] = np.array(hdf.get('cm')[:,block_size*loops::]).transpose()
        create_matrix_fast(self.features,self.hdf_path)
        
        
    def add_new_images(self):
        self.new_image_file = filedialog.askopenfilename(defaultextension=".txt")
        with open(self.new_image_file, 'r') as file:
            new_images = file.readlines()
            
        num_new_images = len(new_images)
        with h5py.File(self.hdf_path, 'a') as hdf:
            hdf['thumbnail_images'].resize((len(hdf['thumbnail_images']) + num_new_images, 100, 100, 3))

        new_features = self.feature_extraction(self.neuralnet,new_images,new_imgs=True)
        block_size = 1000
        len_f = len(self.features)
        len_nf = len(new_features)
    
        with h5py.File(self.hdf_path, 'a') as hdf:
            hdf['cm'].resize((len_f + len_nf, len_f + len_nf))

            loops_new = math.floor(len_nf/block_size)
        
            # calculate new features correlation
            for loop1 in range(loops_new):
                features_p1 = new_features[loop1*block_size:loop1*block_size + block_size]
                for loop2 in range(loop1, loops_new):
                    features_p2 = new_features[loop2*block_size:loop2*block_size + block_size]
                    cm_p = np.corrcoef(features_p1,features_p2)[block_size::,0:block_size]
                    if loop1 == loop2:
                        for id_1 in range(len(features_p1)):
                            cm_p[id_1,id_1]=0
        
                    hdf['cm'][len_f+loop2*block_size:len_f+loop2*block_size + block_size, len_f+loop1*block_size:len_f+loop1*block_size + block_size] = cm_p
        
                    if loop1 != loop2:
                        hdf['cm'][len_f+loop1*block_size:len_f+loop1*block_size + block_size,len_f+loop2*block_size:len_f+loop2*block_size + block_size] = cm_p.transpose()
        
            remaining_new = len_nf - block_size * loops_new
            if remaining_new > 0:
                features_p1 = new_features[block_size*loops_new::]
                for loop2 in range(loops_new):
                    features_p2 = new_features[loop2*block_size:loop2*block_size + block_size]
                    cm_p = np.corrcoef(features_p2,features_p1)[0:block_size,block_size::]
                    hdf['cm'][len_f+loop2*block_size:len_f+loop2*block_size + block_size, len_f+block_size*loops_new::] = cm_p
        
                cm_p = np.corrcoef(features_p1)
                for id_1 in range(len(cm_p)):
                    cm_p[id_1,id_1]=0
        
                hdf['cm'][len_f+block_size*loops_new::,len_f+block_size*loops_new::] = cm_p
                hdf['cm'][len_f+block_size*loops_new::,:] = np.array(hdf.get('cm')[:,len_f+block_size*loops_new::]).transpose()

            loops = math.floor(len_f/block_size)
            # calculate correlation between old and new features
            for loop1 in range(loops):
                features_p1 = self.features[loop1*block_size:loop1*block_size + block_size]
                for loop2 in range(loops_new):
                    features_p2 = new_features[loop2*block_size:loop2*block_size + block_size]
                    cm_p = np.corrcoef(features_p1,features_p2)[block_size::,0:block_size]
            
                    hdf['cm'][loop1*block_size:loop1*block_size + block_size, len_f+loop2*block_size:len_f+loop2*block_size + block_size] = cm_p
                    hdf['cm'][len_f+loop2*block_size:len_f+loop2*block_size + block_size, loop1*block_size:loop1*block_size + block_size] = cm_p.transpose()
            
            remaining_old_new = len_f - block_size * loops
            if remaining_old_new > 0:
                features_p1 = self.features[block_size*loops::]
                for loop2 in range(loops_new):
                    features_p2 = new_features[loop2*block_size:loop2*block_size + block_size]
                    cm_p = np.corrcoef(features_p1,features_p2)[block_size::,0:block_size]
                    hdf['cm'][block_size*loops::,len_f+loop2*block_size:len_f+loop2*block_size + block_size] = cm_p
                    hdf['cm'][len_f+loop2*block_size:len_f+loop2*block_size + block_size,block_size*loops::] = cm_p.transpose()
            
            remaining_new_old = len_nf - block_size * loops_new
            if remaining_new_old > 0:
                features_p2 = new_features[block_size*loops_new::]
                for loop1 in range(loops):
                    features_p1 = self.features[loop1*block_size:loop1*block_size + block_size]
                    cm_p = np.corrcoef(features_p1,features_p2)[0:block_size,block_size::]
                    hdf['cm'][loop1*block_size:loop1*block_size + block_size,len_f+block_size*loops_new::] = cm_p
                    hdf['cm'][len_f+block_size*loops_new::,loop1*block_size:loop1*block_size + block_size] = cm_p.transpose()
        
        self.features = np.vstack((self.features, new_features))
        self.im_list = self.im_list + new_images

        self.communication_label.configure(text = 'Added new images. You can now start clustering the images by pressing Cluster images. You may also want to save now that the features are calculated')
        self.communication_label['background'] = '#FFFFFF'

            

    def remove_images(self):
        remove_image_file = filedialog.askopenfilename(defaultextension=".txt")
        with open(remove_image_file, 'r') as file:
            images_to_remove = file.readlines()

        images_to_remove_normalized = [os.path.normpath(path) for path in images_to_remove]
        im_list_normalized = [os.path.normpath(path) for path in self.im_list]
        
        # Find indices to remove
        indices_to_remove = [i for i, img in enumerate(im_list_normalized) if img in images_to_remove_normalized]
        
        # Remove from self.im_list
        self.im_list = [img for i, img in enumerate(self.im_list) if i not in indices_to_remove]
        self.features = [img for i, img in enumerate(self.features) if i not in indices_to_remove]
        for key in self.theBuckets.keys():
            self.theBuckets[key] = [index for index in self.theBuckets[key] if index not in indices_to_remove]
        with h5py.File(self.hdf_path, 'a') as hdf:
            old_ds = hdf['cm']
    
            # Calculate new shape after removing rows and columns
            new_shape = (old_ds.shape[0] - len(indices_to_remove), old_ds.shape[1] - len(indices_to_remove))
            
            new_ds = hdf.create_dataset('cm_new', new_shape, dtype=old_ds.dtype)
            
            # Indices to keep
            indices_to_keep = [i for i in range(old_ds.shape[0]) if i not in indices_to_remove]
            
            for i, old_i in enumerate(indices_to_keep):
                row_data = old_ds[old_i]
                row_data = np.delete(row_data, indices_to_remove)  # remove columns
                new_ds[i] = row_data
        
            del hdf['cm']  # delete the old dataset
        
            # Once the old dataset is deleted, create a new dataset with the desired name 'cm' and copy data from 'cm_new' to 'cm'
            hdf['cm'] = new_ds[...]
            del hdf['cm_new']  # Now you can delete the 'cm_new' dataset
        self.communication_label.configure(text = 'Images removed. Make sure to save, or the save file will be broken.')
        self.communication_label['background'] = '#FFFFFF'

        
        

    def create_matrix2(self,focusfeatures,features,distance_metric):
            focusfeatures = np.squeeze(np.expand_dims(focusfeatures,0))
            features = np.squeeze(np.expand_dims(features,0))
            features_t = np.transpose(features)
            focusfeatures_t = np.transpose(focusfeatures)
            cm = []                
            sumX = sum(features_t)
            focussumX = sum(focusfeatures_t)
            sumsquareX = sum(features_t**2)            
            feat0 = focusfeatures
            sumXY = np.dot(feat0,features_t)
            r = features.shape[1]*sumXY - focussumX*sumX 
            s = ((features.shape[1] * sumsquareX) - sumX**2)
            t = 1./((s[0]*s)**0.5)
            u = r * t
            cm.append(u)
            cm = np.asarray(cm)
            return cm
        
    #function to cluster all images based on the correlation score, as specified in []
    def cluster_images(self):
        self.communication_label.configure(text='Calculating the clusters. This may take several minutes, depending on the number of images.')
        self.communication_label['background'] = '#99CCFF'
        self.communication_label.update()
        try:
            self.threshold = float(self.threshold_entry.get())
        except ValueError:
            self.threshold = 0.5
        if self.threshold < 0:
            self.threshold = 0.5
        if self.threshold > 1:
            self.threshold = 0.5
        
        self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'clustering images with threshold '+ str(self.threshold))
        
        def clustering(cm_path,threshold):

            cluster_ind = []
            clustered_images = []
            with h5py.File(cm_path, 'r') as hdf:
                cm_len = hdf['cm'].shape[1]
                cm_sum = np.zeros((cm_len,1))
                for mm in range(cm_len):
                    #warnings.filterwarnings("ignore")
                    # cm_sum[mm,0] = len(np.where(np.array(hdf.get('cm')[mm,:])>threshold)[0])
                    cm_sum[mm,0] = np.count_nonzero(np.array(hdf.get('cm')[mm,:]) > threshold)
                size = len(clustered_images)
                weight2 = 1
                m = cm_len
                tt = 0
                while size < cm_len:
                    if tt > 0 and threshold > 0.1:
                        threshold = threshold-0.1
                        cm_len = hdf['cm'].shape[1]
                        cm_sum = np.zeros((cm_len,1))
                        for mm in range(cm_len):
                            # cm_sum[mm,0] = len(np.where(np.array(hdf.get('cm')[mm,:])>threshold)[0])
                            cm_sum[mm,0] = np.count_nonzero(np.array(hdf.get('cm')[mm,:]) > threshold)
                    elif tt > 0 and threshold <= 0.1:
                        remains = np.arange(0,m)
                        remains[np.asarray(clustered_images)] = -100            
                        final_column = remains[remains > 0]
                        cluster_ind.append(final_column)
                        # cm[final_column,:] = -100
                        # cm[:,final_column] = -100
                        break
                    
                    tt = 1
                    new_cluster = []
                    #while np.sum(cm_bool) > 0:
                    while np.sum(cm_sum) - len(new_cluster)  > 0:
                        tg = 0
                        for g in range(0,len(cluster_ind)):
                            tg = tg + len(cluster_ind[g])
                        #cm_sum = np.sum(cm_bool,1)
                        cm_most = np.argmax(cm_sum)
                        column1 = np.array(hdf.get('cm')[cm_most,:])
                        try:
                            column1[np.asarray(clustered_images)] = -100
                        except IndexError:
                            pass                                            
                        new_cluster = []
                        column_max = bn.nanargmax(column1)
                        if column1[column_max] > threshold:
                            column2 = np.array(hdf.get('cm')[column_max,:])
                            try:
                                column2[np.asarray(clustered_images)] = -100
                            except IndexError:
                                pass
                                
                            
                            weight1 = 1        
                            new_column = float(weight1) / (weight1+weight2) * column1 + float(weight2)/(weight1+weight2)*column2
                            new_cluster.append(cm_most)
                            new_cluster.append(column_max)
                            clustered_images.append(cm_most)
                            clustered_images.append(column_max)
                            new_column[new_cluster] = 0
                            while bn.nanmax(new_column) > threshold:
                                weight1 = weight1 + 1
                                column1 = new_column
                                column_max = bn.nanargmax(column1)
                                
                                column2 = np.array(hdf.get('cm')[column_max,:])
                                try:
                                    column2[np.asarray(clustered_images)] = -100
                                except IndexError:
                                    pass
                                new_column = float(weight1) / (weight1+weight2) * column1 + float(weight2)/(weight1+weight2)*column2
                                # cm_bool[column_max,:] = 0
                                # cm_bool[:,column_max] = 0
                                new_cluster.append(column_max)
                                clustered_images.append(column_max)
                                new_column[new_cluster] = 0
                        else:
                            break
                        cluster_ind.append(new_cluster)
                        cm_sum[new_cluster] = 0
                        ## cm[new_cluster,:] = -100
                        ## cm[:,new_cluster] = -100
                        size = len(clustered_images)
            xlength = []
            for q in range(0,len(cluster_ind)):
                xlength.append(len(cluster_ind[q]))
            cluster_indX = np.zeros((max(xlength),len(cluster_ind)))-1        
            for r in range(0,len(cluster_ind)):
                for s in range(0,len(cluster_ind[r])):
                    cluster_indX[s,r] = int(cluster_ind[r][s])
            cluster_indX = cluster_indX.astype(int)                        
                
            return cluster_indX


        self.df = clustering(self.hdf_path,self.threshold)
        #self.df = pd.DataFrame.from_records(self.df)
        self.communication_label.configure(text='Calculated the clusters. You can now start browsing the clusters by Next cluster. You could also preload the images for a smoother browsing experience. Preloading takes a moment initially and uses a bit more system memory.')
        self.communication_label['background'] = '#FFFFFF'
        num_c = 0
        self.num_c = 0
    



    def create_hyperedge_checkboxes(self):
        """Creates a list of checkboxes in self.left_frame for each overlapping hyperedge.
        The first five (default drawn) are checked by default."""
        # Clear any existing checkboxes
        for widget in self.left_frame.winfo_children():
            widget.destroy()

        self.checkbox_vars = {}
        for idx, edge_id in enumerate(self.edge_ids):
            var = BooleanVar()
            # Check the first five by default.
            var.set(True if idx < 5 else False)
            self.checkbox_vars[edge_id] = var
            chk = Checkbutton(
                self.left_frame,
                text=str(edge_id),
                variable=var,
                bg="#555555",
                fg="white",
                activebackground="#555555",
                activeforeground="white",
                selectcolor="#777777",  # Choose a contrasting color
                command=self.redraw_hypergraph  # Update drawing when toggled.
            )
            chk.pack(anchor="w", padx=5, pady=2)


    def hyperedge_preparation(self):
        """Based on the currently selected hyperedge, determine which other hyperedges need to be shown."""
        if self.selected_edge is None:
            self.selected_edge = next(iter(self.hyperedges))
        # Get the images in the selected hyperedge
        self.edge_images = self.hyperedges[self.selected_edge]
        
        overlapping_hyperedges = set()
        for image_id in self.edge_images:
            hyperedges_of_image = self.image_mapping[image_id]
            overlapping_hyperedges.update(hyperedges_of_image)
        
        # Remove the selected hyperedge from the set of overlapping hyperedges
        overlapping_hyperedges.discard(self.selected_edge)
        self.overlapping_hyperedge_names = overlapping_hyperedges
        # Prepare a list to store hyperedge data
        self.overlapping_hyperedges = []
        edge_ids = []
        for hyperedge in overlapping_hyperedges:
            edge_ids.append(hyperedge)
            images_in_hyperedge = self.hyperedges[hyperedge]
            overlapping_images = self.edge_images.intersection(images_in_hyperedge)
            non_overlapping_images = images_in_hyperedge - overlapping_images
            # Create an ordered list of images: shared images first
            ordered_images = list(overlapping_images) + list(non_overlapping_images)
            
            # Convert the ordered images list to a NumPy array
            hyperedge_array = np.array(ordered_images, dtype=int)
            
            # Append the NumPy array to the list
            self.overlapping_hyperedges.append(hyperedge_array)
        
        sorted_indices = sorted(
            range(len(self.overlapping_hyperedges)),
            key=lambda i: len(np.intersect1d(self.overlapping_hyperedges[i], list(self.edge_images))),
            reverse=True
        )
        
        self.overlapping_hyperedges = [self.overlapping_hyperedges[i] for i in sorted_indices]
        self.edge_ids = [edge_ids[i] for i in sorted_indices]

        self.create_hyperedge_checkboxes()

        self.edge_images = np.array(list(self.edge_images), dtype=int)
        hyperedge_features = self.features[np.array(self.edge_images)]
        avg_vector = np.mean(hyperedge_features, axis=0)
        cosine_similarities = cosine_similarity(hyperedge_features, avg_vector.reshape(1, -1)).flatten()
        sorted_indices = np.argsort(-cosine_similarities)
        self.edge_images = self.edge_images[sorted_indices]



    def get_hyperedge_color_mappings(self):
        """
        Creates consistent mappings for hyperedge colors and labels using self.hyperedges.

        Returns:
            color_map: dict mapping hyperedge ID to colors.
            label_map: dict mapping hyperedge ID to labels.
        """
        # Generate consistent colors for hyperedges
        unique_edges = list(self.hyperedges.keys())
        cmap = colormaps["tab10"]
        colors = [mcolors.to_hex(cmap(i / len(unique_edges))) for i in range(len(unique_edges))]
        color_map = {edge: colors[i] for i, edge in enumerate(unique_edges)}
        

        # Generate labels for hyperedges
        label_map = {edge: f"Edge {edge}" for edge in unique_edges}

        return color_map, label_map

    

    def draw_hyper_edges(
        self, H, pos, ax=None, node_radius={}, contain_hyper_edges=False, dr=None, **kwargs
    ):
        """
        Draws the hyperedges with consistent colors using self.hyperedges.

        Args:
            H: The hypergraph.
            pos: Layout positions for nodes and edges.
        """
        # Get color mapping from self.hyperedges
        color_map, _ = self.get_hyperedge_color_mappings()
        
        # Layout for hyperedges
        points = hnx.drawing.rubber_band.layout_hyper_edges(
            H, pos, node_radius=node_radius, dr=dr, contain_hyper_edges=contain_hyper_edges
        )

        # Assign colors to edges
        edge_colors = [color_map[edge] for edge in H.edges]  

        polys = PolyCollection(
            points, edgecolors=edge_colors, facecolors="none", **kwargs
        )

        (ax or plt.gca()).add_collection(polys)
        return polys


    def draw_hyper_edge_labels(self, H, pos, polys, ax=None, **kwargs):
        """
        Draws labels for hyperedges with consistent text from self.hyperedges.

        Args:
            H: The hypergraph.
            pos: Layout positions for nodes and edges.
            polys: PolyCollection of hyperedges.
        """
        # Get label mapping from self.hyperedges
        _, label_map = self.get_hyperedge_color_mappings()

        ax = ax or plt.gca()

        for edge, path in zip(H.edges.keys(), polys.get_paths()):
            label = label_map.get(edge, f"Edge {edge}")

            # Calculate label position (midpoint of longest segment)
            d = ((path.vertices[:-1] - path.vertices[1:]) ** 2).sum(axis=1)
            i = d.argmax()

            x1, x2 = path.vertices[i : i + 2]
            xy = (x1 + x2) / 2

            # Annotate with consistent label
            ax.annotate(
                label,
                xy,
                color="black",  # Label color can differ from edge color
                ha="center",
                va="center",
                **kwargs
            )

    def draw_hypergraph(self, H, pos=None, ax=None, return_pos=False, **kwargs):
        """
        Main drawing function for the hypergraph with consistent styling.
        """
        ax = ax or plt.gca()

        # Generate layout if not provided
        if pos is None:
            pos = hnx.drawing.rubber_band.layout_node_link(H)

        # Draw edges and labels
        polys = self.draw_hyper_edges(H, pos, ax=ax, **kwargs)
        self.draw_hyper_edge_labels(H, pos, polys, ax=ax, **kwargs)

        # Draw nodes
        hnx.drawing.rubber_band.draw_hyper_nodes(H, pos, ax=ax, **kwargs)

        ax.axis("equal")
        ax.axis("off")
        if return_pos:
            return pos


    def display_overlapping_hyperedges(self):
        # Destroy existing widgets in the new window
        for widget in self.bottomright_imex_frame.winfo_children():
            if (hasattr(self, 'canvasG') and widget == self.canvasG.get_tk_widget()) or \
                    (hasattr(self, 'linked') and widget == self.linked):
                # Skip destroying self.canvasG or self.linked if they exist
                continue
            widget.destroy()
        
        # Clear the list of hyperedge_canvases
        self.hyperedge_canvases.clear()
        
        # Create a main canvas and scrollbar in the new window
        self.main_canvas = Canvas(self.bottomright_imex_frame, bg='#555555')
        self.main_canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        
        scrollbar = Scrollbar(self.bottomright_imex_frame, orient=VERTICAL, command=self.main_canvas.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        self.main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create a frame inside the main canvas
        frame = Frame(self.main_canvas, bg='#555555')
        self.main_canvas.create_window((0, 0), window=frame, anchor='nw')
        
        # Update scrollregion when the frame size changes
        frame.bind('<Configure>', lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox('all')))
        
        # Configure rows/columns in the frame so they expand properly
        frame.grid_columnconfigure(0, weight=1)
        # We'll place each hyperedge in consecutive rows
        row_idx = 0
        self.hyperedge_frames = []

        if self.linked.var.get() == False:
            # For each overlapping hyperedge
            for he_id, hyperedge in enumerate(self.overlapping_hyperedges):
                if he_id < 15:
                    # Label for the hyperedge
                    label = Label(frame, text=self.edge_ids[he_id], bg='#0d0933', fg='white', font=('Arial', 14),relief="raised",)
                    label.edge_id = self.edge_ids[he_id]
                    label.bind("<Button-1>", lambda e: self.show_edge_from_side(e.widget.edge_id))
                    label.grid(row=row_idx, column=0, sticky='ew', padx=10, pady=5)
                    row_idx += 1

                    # Frame to hold the canvas and its scrollbar
                    canvas_frame = Frame(frame, bg='#555555')
                    canvas_frame.grid(row=row_idx, column=0, sticky='nsew', padx=10, pady=5)
                    
                    # Allow canvas_frame to expand
                    frame.grid_rowconfigure(row_idx, weight=1)

                    # Canvas for the hyperedge images
                    hyperedge_canvas = Canvas(canvas_frame, bg='#555555', width=800, height=600)
                    hyperedge_canvas.grid(row=0, column=0, sticky='nsew')

                    # Configure canvas_frame to expand
                    canvas_frame.grid_rowconfigure(0, weight=1)
                    canvas_frame.grid_columnconfigure(0, weight=1)

                    hyperedge_canvas.my_tag = he_id
                    # Vertical scrollbar for the hyperedge canvas
                    v_scrollbar = Scrollbar(canvas_frame, orient=VERTICAL, command=hyperedge_canvas.yview)
                    v_scrollbar.grid(row=0, column=1, sticky='ns')

                    hyperedge_canvas.configure(yscrollcommand=v_scrollbar.set)

                    # Bind mouse wheel events to the hyperedge canvas
                    hyperedge_canvas.bind('<Enter>', lambda event: event.widget.focus_set())
                    hyperedge_canvas.bind('<Enter>', self._bound_to_mousewheel_second_window)
                    hyperedge_canvas.bind('<Leave>', self._unbound_to_mousewheel_second_window)
                    hyperedge_canvas.bind('<Button-2>', self.open_image3)
                    hyperedge_canvas.bind('<Button-1>', self.click_select)
                    hyperedge_canvas.bind('<Button-3>', self.click_select_in_overlapping_hyperedges)
                    hyperedge_canvas.bind('<Shift-Button-1>', self.shift_click_select)
                    hyperedge_canvas.bind('<Control-Button-1>', self.ctrl_click_select)
                    hyperedge_canvas.bind('<Double-Button-1>', self.double_click_overview)

                    self.hyperedge_canvases.append(hyperedge_canvas)
                    
                    self.hyperedge_frames.append(canvas_frame)
                    # Get image IDs and display them
                    image_ids = hyperedge
                    image_indices = np.array(list(image_ids), dtype=int)
                    if len(image_indices) > 100:
                        image_indices = image_indices[0:100]
                    self.display_images_on_canvas(self.hyperedge_canvases[-1], image_indices)

                    # Move to the next row for next hyperedge
                    row_idx += 1
                else:
                    break
        elif self.linked.var.get() == True:
            # For each hyperedge in self.scenes, create a canvas and display images
            for he_id, (hyperedge_name, hyperedge_data) in enumerate(self.scenes.items()):
                # hyperedge_name is already the real key (no need to do f"edge_{hyperedge_id}")
                
                # Retrieve all images belonging to the current hyperedge
                if hyperedge_name in self.filtered_hyperedges:
                    images = self.filtered_hyperedges[hyperedge_name]
                else:
                    continue  # Skip if the hyperedge is not found

                # Reorder images
                pvp_set = set(self.previous_visible_points)
                prioritized = [img for img in images if img in pvp_set]
                remaining = [img for img in images if img not in pvp_set]
                reordered_images = prioritized + remaining
                if len(reordered_images) > 100:
                    reordered_images = reordered_images[:100]

                # Label for the hyperedge
                label = Label(frame, text=hyperedge_name, bg='#555555', fg='white', font=('Arial', 14))
                label.grid(row=row_idx, column=0, sticky='ew', padx=10, pady=5)
                row_idx += 1

                # Frame to hold the canvas and its scrollbar
                canvas_frame = Frame(frame, bg='#555555')
                canvas_frame.grid(row=row_idx, column=0, sticky='nsew', padx=10, pady=5)
                frame.grid_rowconfigure(row_idx, weight=1)

                # Canvas for the hyperedge images
                hyperedge_canvas = Canvas(canvas_frame, bg='#555555', width=800, height=600)
                hyperedge_canvas.grid(row=0, column=0, sticky='nsew')

                canvas_frame.grid_rowconfigure(0, weight=1)
                canvas_frame.grid_columnconfigure(0, weight=1)

                hyperedge_canvas.my_tag = he_id

                # Vertical scrollbar for the hyperedge canvas
                v_scrollbar = Scrollbar(canvas_frame, orient=VERTICAL, command=hyperedge_canvas.yview)
                v_scrollbar.grid(row=0, column=1, sticky='ns')
                hyperedge_canvas.configure(yscrollcommand=v_scrollbar.set)

                # Bind mouse wheel events
                hyperedge_canvas.bind('<Enter>', lambda event: event.widget.focus_set())
                hyperedge_canvas.bind('<Enter>', self._bound_to_mousewheel_second_window)
                hyperedge_canvas.bind('<Leave>', self._unbound_to_mousewheel_second_window)
                hyperedge_canvas.bind('<Button-2>', self.open_image3)
                hyperedge_canvas.bind('<Button-1>', self.click_select)
                hyperedge_canvas.bind('<Button-3>', self.click_select_in_overlapping_hyperedges)
                hyperedge_canvas.bind('<Shift-Button-1>', self.shift_click_select)
                hyperedge_canvas.bind('<Control-Button-1>', self.ctrl_click_select)
                hyperedge_canvas.bind('<Double-Button-1>', self.double_click_overview)

                self.hyperedge_canvases.append(hyperedge_canvas)
                self.hyperedge_frames.append(canvas_frame)

                image_indices = np.array(list(reordered_images), dtype=int)
                self.display_images_on_canvas(self.hyperedge_canvases[-1], image_indices)

                # Move to the next row for the next hyperedge
                row_idx += 1


    def display_images_on_canvas(self, canvas, image_indices):
        # Similar setup as in display_images, but use the provided canvas
        self.sharetangles = []
        num_im_row = math.floor(800 / (self.imsize + self.image_distance))  # Assuming canvas width is 800

        # Prepare the canvas scroll region
        total_rows = math.ceil(len(image_indices) / num_im_row)
        canvas.config(scrollregion=(0, 0, 0, total_rows * (self.imsize + self.image_distance)))

        # Clear the canvas
        canvas.delete("all")

        # Initialize a list to store image references
        if not hasattr(canvas, 'image_refs'):
            canvas.image_refs = []
        else:
            # Clear previous references
            canvas.image_refs.clear()

        # Gather images to highlight based on `self.linked.var.get()`
        highlighted_images = set()
        if self.linked.var.get():
            # Collect all images in `self.scenes`
            for images in self.scenes.values():
                highlighted_images.update(images)
        else:
            # Use images in `self.edge_images`
            highlighted_images = set(self.edge_images)

        # Load and display images
        with h5py.File(self.hdf_path, 'r') as hdf:
            for idx, img_idx in enumerate(image_indices):
                load = Image.fromarray(np.array(hdf.get('thumbnail_images')[img_idx], dtype='uint8'))
                render = ImageTk.PhotoImage(load)
                row_ = idx // num_im_row
                column_ = idx % num_im_row
                x_pos = column_ * (self.imsize + self.image_distance) + (self.imsize / 2)
                y_pos = row_ * (self.imsize + self.image_distance) + (self.imsize / 2)
                for_bb = canvas.create_image(x_pos, y_pos, image=render, tags=(img_idx))
                canvas.image_refs.append(render)

                # Draw rectangle if the image is in `highlighted_images`
                if img_idx in highlighted_images:
                    bbox = canvas.bbox(for_bb)
                    rect = canvas.create_rectangle(
                        bbox,
                        outline='yellow',
                        width=2,
                        tags='share_tag'
                    )
                    self.sharetangles.append(rect)
                if hasattr(self, 'selected_image_id') and img_idx == self.selected_image_id:
                    bbox = canvas.bbox(for_bb)
                    rect = canvas.create_rectangle(
                        bbox,
                        outline='dodgerblue',
                        width=3,
                        tags='selected_tag'
                    )




    def display_hyperedges(self):
        self.hyperedge_preparation()
        self.display_images(self.edge_images)
        self.display_overlapping_hyperedges()
        if len(self.represent) != len(self.hyperedges.keys()):
            self.hyperedge_size = []
            self.hyperedge_rep_name = []
            for edge in self.hyperedges:
                edge_imgs = np.asarray(list(self.hyperedges[edge]),dtype=int)
                edge_features = self.features[edge_imgs]
                self.hyperedge_size.append(edge_features.shape[0])
                self.hyperedge_rep_name.append(edge)
                avg_edge = np.mean(edge_features,axis=0).reshape(1, -1)
                corrs = []
                for vec in edge_features:
                    corrs.append(np.corrcoef(avg_edge,vec)[0,1])
                rep_img = edge_imgs[np.argmax(corrs)]
                self.represent.append(rep_img)
        self.display_matrix()
        self.show_edge_view()
        

##############################
###### hypergraph lasso ######
##############################
    def on_key_press(self, event):
        """Called whenever a key is pressed over the figure."""        
        if event.key == 'control':
            self.ctrl_is_held = True

    def on_key_release(self, event):
        """Called whenever a key is released over the figure."""
        if event.key == 'control':
            self.ctrl_is_held = False
            # If a lasso was in progress, disconnect it
            if self.lasso:
                self.lasso.disconnect_events()
                self.lasso = None

    def on_mouse_press(self, event):
        """Called on any mouse press in the figure."""
        # Only create a lasso if user is holding Ctrl, left-button, and no other lasso is active
        if self.ctrl_is_held and event.button == 1 and self.lasso is None:
            self.lasso = LassoSelector(self.ax, onselect=self.on_lasso_select,
                                       props={'color':'red','linewidth':1,'alpha':0.8})
            

    def on_lasso_select(self, verts):
        path = Path(verts)
        self.lasso_indices = []
        for i, (x, y) in enumerate(self.filtered_umap_features):
            if path.contains_point((x, y)):
                if i in self.selected_indices_for_umap:
                    self.lasso_indices.append(i)
        # Then show those images:
        self.display_images(self.lasso_indices)
        
        # data = {}
        # for image in self.lasso_indices:
        #     row_dict = {}   
        #     row_dict['label'] = str(image)  # display image as row label

        #     for edge in self.all_edges:
        #         row_dict[edge] = '1' if image in self.hyperedges[edge] else '0'
            
        #     data[image] = row_dict

        # model = TableModel()
        # model.importDict(data)
        # mview = SyncScrollExample(self.tab1, self.lasso_indices, self.hdf_path, data, model)  # Try bigger row_count
        # mview.grid(row=0, column=0, sticky="nsew")

        # self.tab1.grid_columnconfigure(0, weight=1)
        # self.tab1.grid_rowconfigure(0, weight=1)

        # # Bind the right-click event (if needed) to tab1 (or you can bind it to mview directly)
        # self.tab1.bind('<Button-3>', self.click_select_in_matrix)
        # Clean up lasso
        self.lasso.disconnect_events()
        self.lasso = None
##############################
#### hypergraph lasso end ####
##############################


    # def rename_hyperedge(self):
    #     """
    #     Renames the hyperedge selected in the listbox to the text entered in a popup dialog
    #     and updates self.hyperedges and self.image_mapping accordingly.
    #     """
    #     selected_index = self.categories.curselection()
    #     if len(selected_index) != 1:
    #         return

    #     old_name = self.categories.get(selected_index)
        
    #     # Ask the user for a new name using a popup dialog.
    #     new_name = simpledialog.askstring("Rename Hyperedge", f"Change name of {old_name} to:")
    #     if new_name is None:
    #         # User pressed Cancel
    #         return

    #     new_name = new_name.strip()
    #     if not new_name:
    #         return

    #     if new_name != old_name:
    #         existing_names = self.categories.get(0, "end")  # all names in the Listbox
    #         if new_name in existing_names:
    #             return

    #     self.categories.delete(selected_index)
    #     self.categories.insert(selected_index, new_name)

    #     if old_name in self.hyperedges:
    #         self.hyperedges[new_name] = self.hyperedges.pop(old_name)
    #     else:
    #         return

    #     for i_idx, edge_set in self.image_mapping.items():
    #         if old_name in edge_set:
    #             edge_set.remove(old_name)
    #             edge_set.add(new_name)
    def rename_hyperedge(self):
        """
        Renames the hyperedge selected in the listbox to the text entered in a popup dialog
        and updates self.hyperedges and self.image_mapping accordingly.
        """
        selected_index = self.categories.curselection()
        if len(selected_index) != 1:
            return

        old_name = self.categories.get(selected_index)

        new_name = simpledialog.askstring("Rename Hyperedge", f"Change name of {old_name} to:")
        if new_name is None:
            return  # Cancelled

        new_name = new_name.strip()
        if not new_name or new_name == old_name:
            return

        existing_names = self.categories.get(0, "end")
        if new_name in existing_names:
            return

        # Update Listbox
        self.categories.delete(selected_index)
        self.categories.insert(selected_index, new_name)

        if old_name not in self.hyperedges:
            return

        # Reconstruct OrderedDict with renamed key in same position
        new_hyperedges = OrderedDict()
        for key, value in self.hyperedges.items():
            if key == old_name:
                new_hyperedges[new_name] = value
            else:
                new_hyperedges[key] = value
        self.hyperedges = new_hyperedges

        # Update image mapping
        for edge_set in self.image_mapping.values():
            if old_name in edge_set:
                edge_set.remove(old_name)
                edge_set.add(new_name)        

    def remove_selected_image_from_hyperedge(self):
        if self.selected_images == None:
            return
        if self.selected_edge == None: 
            return
        for i_idx in self.selected_images:
            im_idx = int(i_idx.split('_')[1])
            self.remove_image_from_hyperedge(self.selected_edge, im_idx)
        self.display_hyperedges()

    def remove_image_from_hyperedge(self, hyperedge_name, i_idx):
        if hyperedge_name in self.hyperedges:
            self.hyperedges[hyperedge_name].discard(i_idx)  # Use discard to avoid errors if i_idx is not in the set

            if len(self.hyperedges[hyperedge_name]) == 0:
                del self.hyperedges[hyperedge_name]

        if i_idx in self.image_mapping:
            self.image_mapping[i_idx].discard(hyperedge_name)

            if len(self.image_mapping[i_idx]) == 0:
                del self.image_mapping[i_idx]
        
        self.selected_edge = self.current_edge
        



    def add_selected_image_to_hyperedge(self):
        """
        Adds the selected images to the currently selected hyperedge.
        """
        self.hyperedges[self.selected_edge]
        self.selected_edge = self.categories.get(self.categories.curselection()) 
        if self.selected_images is None:
            return
        if self.selected_edge is None:
            return
        for i_idx in self.selected_images:
            im_idx = int(i_idx.split('_')[1])
            self.add_image_to_hyperedge(self.selected_edge, im_idx)


    def add_image_to_hyperedge(self, hyperedge_name, i_idx):
        """
        Adds the given image index (i_idx) to the hyperedge (hyperedge_name)
        and updates the image mapping accordingly.
        """

        self.hyperedges[hyperedge_name].add(i_idx)

        if i_idx not in self.image_mapping:
            self.image_mapping[i_idx] = set()
        self.image_mapping[i_idx].add(hyperedge_name)


    def add_hyperedge(self):
        """
        Adds a new hyperedge to the hypergraph using a popup dialog.
        Updates the Listbox (self.categories) and the self.hyperedges dictionary.
        """
        new_name = simpledialog.askstring("Add Hyperedge", "Enter new hyperedge name:")
        if new_name is None:
            # User cancelled the dialog.
            return

        new_name = new_name.strip()
        if not new_name:
            return

        existing_names = self.categories.get(0, "end")
        if new_name in existing_names:
            return

        self.categories.insert("end", new_name)
        self.hyperedges[new_name] = set()

    def delete_hyperedge(self):
        """
        Deletes the currently selected hyperedge from self.hyperedges, and removes it from the Listbox.
        Any images that become hyperedge-less (i.e., are not in any other hyperedge)
        get moved into 'orphaned_images'.
        """
        selection = self.categories.curselection()
        if len(selection) != 1:
            return  # No single hyperedge selected

        old_edge_name = self.categories.get(selection[0])
        if old_edge_name not in self.hyperedges:
            return  # Edge doesn't exist or was removed

        # For each image in that hyperedge, remove the edge from image_mapping
        orphan_candidates = []
        for img_idx in self.hyperedges[old_edge_name]:
            if img_idx in self.image_mapping:
                # Remove this edge from the image's set of hyperedges
                self.image_mapping[img_idx].discard(old_edge_name)
                # If it's now in no hyperedge, it is orphaned
                if len(self.image_mapping[img_idx]) == 0:
                    orphan_candidates.append(img_idx)
        
        # Remove the hyperedge from the dictionary
        del self.hyperedges[old_edge_name]

        # Remove from the Listbox
        self.categories.delete(selection[0])

        # Move newly orphaned images into the 'orphaned_images' hyperedge
        if orphan_candidates:
            self.ensure_orphaned_images_exists()  # see helper method below
            for img_idx in orphan_candidates:
                self.hyperedges['orphaned_images'].add(img_idx)
                self.image_mapping[img_idx].add('orphaned_images')
        
    def ensure_orphaned_images_exists(self):

        if 'orphaned_images' not in self.hyperedges:
            self.hyperedges['orphaned_images'] = set()
            existing_names = self.categories.get(0, "end")
            if 'orphaned_images' not in existing_names:
                self.categories.insert("end", 'orphaned_images')
    
    def on_shift_right_click(self,event):
        widget = event.widget
        print("You clicked on:", widget)
        # You can also get more details, e.g., widget class
        print("Widget class:", widget.winfo_class())



    def show_matrix(self):
        # self.matrixWindow.bind_all("<Shift-Button-3>", self.on_shift_right_click)
   
        data = {}
        for image in self.images_in_selected:
            row_dict = {}   
            row_dict['label'] = str(image)  # display image as row label

            for edge in self.all_edges:
                row_dict[edge] = '1' if image in self.hyperedges[edge] else '0'
            
            data[image] = row_dict

        # Create a model from this data
        model = TableModel()
        model.importDict(data)
        mview = SyncScrollExample(self.tab1, self.images_in_selected, self.hdf_path, data, model)  # Try bigger row_count
        mview.grid(row=0, column=0, sticky="nsew")

        self.tab1.grid_columnconfigure(0, weight=1)
        self.tab1.grid_rowconfigure(0, weight=1)

        # Bind the right-click event (if needed) to tab1 (or you can bind it to mview directly)
        self.tab1.bind('<Button-3>', self.click_select_in_matrix)
        # if self.confusion_matrix == None:
        for widget in self.tab2.winfo_children():
            widget.destroy()

        
        hyperedge_images = {}
        for i, edge_name in enumerate(self.hyperedges.keys()):
            print(edge_name)      
            hyperedge_images[edge_name] = self.represent[i]#list(self.hyperedges[edge_name])[0]
        print(hyperedge_images)
        self.cell_button = Button(
            self.tab2,
            text="Show hyperedges",
            command=self.display_matrix_selection
        )
        self.cell_button.pack()


        # self.confusion_matrix = ConfusionMatrixWithImages(self.tab2, self.hyperedges, hyperedge_images,self.hdf_path)
        # self.confusion_matrix.pack(fill='both', expand=True)
        # self.confusion_matrix.bg='#555555'
        # self.confusion_matrix.bgcolor='#555555'
        
        self.canvas_for_matrix = Canvas(self.tab2, bg='#555555')
        self.canvas_for_matrix.pack(side=TOP, fill=BOTH, expand=True)

        # Create vertical and horizontal scrollbars and attach them to the canvas:
        v_scrollbar = Scrollbar(self.tab2, orient=VERTICAL, command=self.canvas_for_matrix.yview)
        v_scrollbar.pack(side=RIGHT, fill=Y)
        h_scrollbar = Scrollbar(self.tab2, orient=HORIZONTAL, command=self.canvas_for_matrix.xview)
        h_scrollbar.pack(side=BOTTOM, fill=X)

        # Configure the canvas to use the scrollbars:
        self.canvas_for_matrix.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Create a frame inside the canvas which will hold your confusion matrix:
        self.container_for_matrix = Frame(self.canvas_for_matrix, bg='#555555')

        # Add the frame to the canvas as a window item:
        self.canvas_for_matrix.create_window((0, 0), window=self.container_for_matrix, anchor="nw")

        # Now create your ConfusionMatrixWithImages inside the container:
        self.confusion_matrix = ConfusionMatrixWithImages(self.container_for_matrix, self.hyperedges, hyperedge_images, self.hdf_path, bg='#555555')
        self.confusion_matrix.pack(fill='both', expand=True)

        # Update the canvas's scrollable region whenever the container frame is resized:
        def update_scroll_region(event):
            self.canvas_for_matrix.configure(scrollregion=self.canvas_for_matrix.bbox("all"))

        self.container_for_matrix.bind("<Configure>", update_scroll_region)
        def _on_mousewheel(event):
    # For Windows, event.delta is ±120 typically
            self.canvas_for_matrix.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.canvas_for_matrix.bind_all("<MouseWheel>", _on_mousewheel)



    def display_matrix_selection(self):
        row_index = self.confusion_matrix.table.getSelectedRow()
        col_index = self.confusion_matrix.table.getSelectedColumn()

        if row_index is not None and col_index is not None:
            row_key = list(self.confusion_matrix.table.model.rorder)[row_index]
            col_key = list(self.confusion_matrix.table.model.columnOrder)[col_index]
            
            # Retrieve the images lists for each hyperedge.
            images_row = self.hyperedges.get(row_key, [])
            images_col = self.hyperedges.get(col_key, [])
            
            # Combine the two lists into a union (removes duplicates).
            self.common_images  = list(set(images_row) & set(images_col))
            print(len(self.common_images ))
            self.display_images(np.array(self.common_images ))
        # confusion_matrix.grid(row=0, column=0, sticky="nsew")

    def display_matrix(self):
        # Clear previous contents from the matrixWindow
        for widget in self.tab1.winfo_children():
            widget.destroy()

        all_edges = self.hyperedges.keys() # [e for e in sorted(self.hyperedges.keys())]
        self.all_edges = all_edges
        self.images_in_selected = self.edge_images # images_in_selected
        if len(self.images_in_selected) > 500:
            self.images_in_selected = self.images_in_selected[0:500]
        self.show_matrix()


    def show_representative_overview(self):
        if len(self.represent) != len(self.hyperedges.keys()):
            self.hyperedge_size = []
            self.hyperedge_rep_name = []
            for edge in self.hyperedges:
                edge_imgs = np.asarray(list(self.hyperedges[edge]),dtype=int)
                edge_features = self.features[edge_imgs]
                self.hyperedge_size.append(edge_features.shape[0])
                self.hyperedge_rep_name.append(edge)
                avg_edge = np.mean(edge_features,axis=0).reshape(1, -1)
                corrs = []
                for vec in edge_features:
                    corrs.append(np.corrcoef(avg_edge,vec)[0,1])
                rep_img = edge_imgs[np.argmax(corrs)]
                self.represent.append(rep_img)
        
        self.oview = 1
        self.display_images(self.represent)
        

            # df = self.df + 1
            # nonzeros = df.astype(bool).sum(axis=0)
            # df = None
            # avg_feat_vec = []
            # self.cluster_to_vector = []
            # self.represent = []
            # for i in range(0,self.df.shape[1]):
            #     avg_feat = np.zeros([nonzeros[i],np.size(self.features,1)])
            #     for j in range(0,nonzeros[i]):
            #         avg_feat[j,0:np.size(self.features,1)] = self.features[self.df[j][i]]
            #     avg_feat_vec.append(np.mean(avg_feat,axis=0))
            #     calc_corr = []
            #     for k in range(0,nonzeros[i]):
            #         cc = np.corrcoef(avg_feat_vec[i],avg_feat[k])
            #         calc_corr.append(cc[0][1])
            #     self.represent.append(self.df[np.argmax(calc_corr),i])
            #     self.cluster_to_vector.append(calc_corr)
            
            
        # self.log.append(time.strftime("%H:%M:%S", time.gmtime())+ ' '+'showing overview')
        # try:
        #     del self.ind_for_overview
        #     del self.xsorted
        #     del self.xsortedimtags
        # except AttributeError:
        #     pass
        # self.bucketDisp = 0
        # self.c.delete("all")
        # self.im_numX = []
        # self.imagex = []
        # self.oview = 1
        # self.display_images(self.represent,input_origin='overview')
        # self.communication_label.configure(text='Now showing the overview.')
        # self.communication_label['background'] = '#99CCFF'





    def recreate_tsne(self):
        modelu = umap.UMAP(
        n_neighbors=200,
        min_dist=0.1,
        n_components=2,
        metric='euclidean'
        )

        self.x_test = modelu.fit_transform(self.features)
        
        self.X_embed = copy.deepcopy(self.x_test)
        self.X_embed[:,0] = self.X_embed[:,0] + abs(np.min(self.X_embed[:,0]))
        self.X_embed[:,1] = self.X_embed[:,1] + abs(np.min(self.X_embed[:,1]))
        self.X_embed[:,0] = self.X_embed[:,0]/np.max(self.X_embed[:,0])
        self.X_embed[:,1] = self.X_embed[:,1]/np.max(self.X_embed[:,1])
    

    def create_tsne(self):
        self.log.append(time.strftime("%H:%M:%S", time.gmtime()) + ' ' + 'Showing or creating umap')
        self.tsne_squares = None
    
        def tsne_click(event):
            self.log.append(time.strftime("%H:%M:%S", time.gmtime()) + ' ' + 'display image from umap selection')
            self.tsneclick += 1
            if self.tsneclick == 2:
                self.evex_tsne2 = self.canvas_tsne.canvasx(event.x)
                self.evey_tsne2 = self.canvas_tsne.canvasy(event.y)
                self.tsneclick = 0
                if self.tsne_squares is not None:
                    self.canvas_tsne.delete(self.tsne_squares)
                self.tsne_squares = self.canvas_tsne.create_rectangle(self.evex_tsne1, self.evey_tsne1, self.evex_tsne2, self.evey_tsne2)
    
                xmin = self.evex_tsne1 / self.canvas_tsne.winfo_width()
                xmax = self.evex_tsne2 / self.canvas_tsne.winfo_width()
                ymin = 1 - (self.evey_tsne1 / self.canvas_tsne.winfo_height())
                ymax = 1 - (self.evey_tsne2 / self.canvas_tsne.winfo_height())
    
                X_images = np.where((self.X_embed[:, 0] > np.min((xmin, xmax))) &
                                    (self.X_embed[:, 0] < np.max((xmin, xmax))) &
                                    (self.X_embed[:, 1] > np.min((ymin, ymax))) &
                                    (self.X_embed[:, 1] < np.max((ymin, ymax))))
    
                self.display_images(X_images[0])
            else:
                self.evex_tsne1 = self.canvas_tsne.canvasx(event.x)
                self.evey_tsne1 = self.canvas_tsne.canvasy(event.y)
    
        if len(self.x_test) < 1:
            modelu = umap.UMAP()
            self.x_test = modelu.fit_transform(self.features)
    
        self.X_embed = copy.deepcopy(self.x_test)
        self.X_embed[:, 0] = self.X_embed[:, 0] + abs(np.min(self.X_embed[:, 0]))
        self.X_embed[:, 1] = self.X_embed[:, 1] + abs(np.min(self.X_embed[:, 1]))
        self.X_embed[:, 0] = self.X_embed[:, 0] / np.max(self.X_embed[:, 0])
        self.X_embed[:, 1] = self.X_embed[:, 1] / np.max(self.X_embed[:, 1])
    
        # Prepare data for plotting
        data = {
            'x': self.X_embed[:, 0],
            'y': self.X_embed[:, 1],
            'category': ['Not in bucket'] * len(self.X_embed)
        }
    
        t = list(chain.from_iterable(self.theBuckets.values()))
        xx = list(np.arange(0, len(self.im_list)))
        not_in_bucket = list(sorted(set(np.arange(0, len(self.im_list))) - set(t), key=xx.index))
        in_bucket = t
    
        for idx in not_in_bucket:
            data['category'][idx] = 'Not in bucket'
        for idx in in_bucket:
            data['category'][idx] = 'In bucket'
        for key, data_list in self.theBuckets.items():
            for idx in data_list:
                data['category'][idx] = key
    
        df = pd.DataFrame(data)
    
        # Create a color map for buckets
        unique_buckets = list(set(data['category']) - {'Not in bucket'})
        color_map = {bucket: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for bucket in unique_buckets}
        color_map['Not in bucket'] = (0, 0, 0)  # Black for points not in any bucket
    
        # Create an image with PIL
        img_width = int(self.screen_width / 2)
        img_height = self.screen_height
        image = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
    
        # Draw the points with assigned colors
        for i, row in df.iterrows():
            x = int(row['x'] * img_width)
            y = int((1 - row['y']) * img_height)  # Invert the y-coordinate
            color = color_map[row['category']]
            draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
    
        # Convert PIL image to Tkinter format
        render_tsne = ImageTk.PhotoImage(image)
    
        # Reuse the canvas if it exists, otherwise create a new one
        if hasattr(self, 'canvas_tsne'):
            self.canvas_tsne.create_image(0, 0, anchor='nw', image=render_tsne)
            self.canvas_tsne.image = render_tsne  # Keep a reference to avoid garbage collection
        else:
            self.canvas_tsne = Canvas(self.newWindow, bg='#555544', bd=0, highlightthickness=0, width=img_width + 1, height=img_height + 1)
            self.canvas_tsne.place(x=self.screen_width / 2, y=100)
            self.canvas_tsne.create_image(0, 0, anchor='nw', image=render_tsne)
            self.canvas_tsne.image = render_tsne  # Keep a reference to avoid garbage collection
            self.canvas_tsne.bind("<Button-1>", tsne_click)

    # function to select video folder that you want to add to your clustering results.
    def select_video_foldere(self):
        self.selected_video_folder = filedialog.askdirectory()   #this will make the file directory a string
        if len(self.selected_video_folder) > 0:
            self.vid_list = glob.glob(self.selected_video_folder + '/**/*.mp4', recursive=True)
            self.vid_list.extend(glob.glob(self.selected_video_folder + '/**/*.MP4', recursive=True))
            self.vid_list.extend(glob.glob(self.selected_video_folder + '/**/*.webm', recursive=True))
            self.vid_list = list(set(self.vid_list))
            self.communication_label.configure(text='Continue by pressing the Add video to clusters. Found ' + str(len(self.vid_list)) + ' videos in ' + self.selected_video_folder)
            #self.sel_folder.set('found ' + str(len(self.vid_list)) + ' videos in ' + self.selected_video_folder)


    
    # redo and undo DISPLAY IMAGES ONLY, not actions such as adding to bucket.
    def ctrlzprevious(self):
        self.ctrlzused = 1
        if self.czcurrent > -10:
            self.czcurrent -= 1
        cluster = self.controlZ[self.czcurrent]
        self.display_images(cluster)
        # if self.czcurrent < -1 and len(self.controlZ) == 10:
        #     self.czcurrent += 1
        print(self.czcurrent)
        
    def ctrlznext(self):
        self.ctrlzused = 1
        if self.czcurrent < -1:
            self.czcurrent += 1
        cluster = self.controlZ[self.czcurrent]
        self.display_images(cluster)
        # if self.czcurrent < -10 and len(self.controlZ) == 10:
        #     self.czcurrent += - 1
        print(self.czcurrent)


    
root = Tk()
root.geometry("1900x700")
root.configure(background='#555555')
app = Application(root)
root.mainloop()  
