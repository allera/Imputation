#! /usr/bin/env python
#import imp


from __future__ import print_function
#from future import standard_library
#standard_library.install_aliases()

import warnings
warnings.filterwarnings("ignore")
import sys
import os

flica_toolbox_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(flica_toolbox_path))




 
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import *


#~~~~ FUNCTIONS~~~~

def data_folder():
    global missing_path
    missing_path=askopenfilename()#askdirectory()
    entry.delete(0, END)
    entry.insert(0, missing_path)
    return missing_path

def def_out_dir():
    global out_dir
    out_dir = askdirectory()
    entry5_2.delete(0, END)
    entry5_2.insert(0, out_dir)
    return out_dir


def def_fslpath():
    global fslpath
    fslpath = askdirectory()
    entry5_4_2.delete(0, END)
    entry5_4_2.insert(0, fslpath)
    return fslpath



def gui_calls_imputation999():
    #import pdb;pdb.set_trace()
    #python imputation999.py -savedir '/Users/alblle/Desktop/Imputation_Output2' Tab1.csv
    print(flica_toolbox_path+"/imputation999.py -savedir " + out_dir + '' + missing_path)

    os.system(flica_toolbox_path+"/imputation999.py -savedir " + out_dir + ' ' + missing_path)
    return 1
    


def callback3():
    global out_dir
    out_dir = entry5_2.get()
    entry5_2.delete(0,END)
    entry5_2.insert(0,out_dir)
    return out_dir


root = Tk() # create a top-level window
 
master = Frame(root, name='master') # create Frame in "root"
master.pack(fill=BOTH) # fill both sides of the parent
 
root.title('Imputation 999') # title for top-level window
# quit if the window is deleted
root.protocol("WM_DELETE_WINDOW", master.quit)
 
nb = Notebook(master, name='nb') # create Notebook in "master"
nb.pack(fill=BOTH, padx=2, pady=3) # fill "master" but pad sides
 
#-->INPUT DATA TAB
f1 = Frame(nb, width=600, height=250)
f1.pack(fill=X)
nb.add(f1, text="Input options & run Imputation") # add tab to Notebook

#folder_path = StringVar
Label(f1,text="Select input csv file with missing values coded as 999").grid(row=0, column=0, sticky='e')
entry = Entry(f1, width=50)#, textvariable=missing_path)
entry.grid(row=0,column=1,padx=2,pady=2,sticky='we',columnspan=25)
Button(f1, text="Browse", command=data_folder).grid(row=0, column=27, sticky='ew', padx=8, pady=4)

Label(f1,text="Select output directory").grid(row=1, column=0, sticky='e')
entry5_2 = Entry(f1, width=50)#, textvariable=folder_path)
entry5_2.grid(row=1,column=1,padx=2,pady=2,sticky='we',columnspan=25)
Button(f1, text="Browse", command=def_out_dir).grid(row=1, column=27, sticky='ew', padx=8, pady=4)
Button(f1, text="save", command=callback3).grid(row=1, column=40, sticky='ew', padx=8, pady=4)



#Label(f1,text="Noise estimation").grid(row=4, column=0, sticky='e')
#Button(f1, text="Modality wise", command=noise_est_opts).grid(row=4, column=10, sticky='ew', padx=8, pady=4)
#Button(f1, text="Modality & subject wise", command=noise_est_opts2).grid(row=4, column=20, sticky='ew', padx=8, pady=4)



Button(f1, text="Impute my data please", command=gui_calls_imputation999).grid(row=8, column=14, sticky='ew', padx=8, pady=4)
Button(f1, text="KILL GUI", command=master.quit).grid(row=10, column=14, sticky='ew', padx=8, pady=4)


#<--INPUT DATA TAB







 
# start the app
if __name__ == "__main__":
    master.mainloop() # call master's Frame.mainloop() method.
    #root.destroy() # if mainloop quits, destroy window

