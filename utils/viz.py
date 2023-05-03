import numpy as np 
import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl 


class viz:
    '''Define the default visualize configure
    '''
    # -----------  Palette 1 -------------

    dBlue   = np.array([ 56,  56, 107]) / 255
    Blue    = np.array([ 46, 107, 149]) / 255
    lBlue   = np.array([241, 247, 248]) / 255
    lBlue2  = np.array([166, 201, 222]) / 255
    Green   = np.array([  8, 154, 133]) / 255
    lGreen  = np.array([242, 251, 238]) / 255
    dRed    = np.array([108,  14,  17]) / 255
    Red     = np.array([199, 111, 132]) / 255
    lRed    = np.array([253, 237, 237]) / 255
    lRed2   = np.array([254, 177, 175]) / 255
    dYellow = np.array([129, 119,  14]) / 255
    Yellow  = np.array([220, 175, 106]) / 255
    lYellow2= np.array([166, 201, 222]) / 255
    lYellow = np.array([252, 246, 238]) / 255
    Purple  = np.array([108,  92, 231]) / 255
    ocGreen = np.array([ 90, 196, 164]) / 255
    Gray    = np.array([163, 161, 165]) / 255

    Palette = [Blue, Red, Yellow, ocGreen, Purple, Gray]


    # -----------  Colormap ------------- 

    BluePalette   = [dBlue, Blue, lBlue]
    RedPalette    = [dRed, Red, lRed]
    YellowPalette = [dYellow, Yellow, lYellow]
    RedsMap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'vizReds',    [lRed, dRed])
    mixMap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'vizMix',   [np.array([1]*3), lBlue, Blue, np.array([0]*3)])

    @staticmethod
    def get_style(): 
        # Larger scale for plots in notebooks
        sns.set_context('talk')
        sns.set_style("ticks", {'axes.grid': False})
        mpl.rcParams['axes.spines.right']  = False
        mpl.rcParams['axes.spines.top']    = False
        # Character
        # mpl.rcParams['font.family']        = 'sans-serif'
        # mpl.rcParams['font.sans-serif']    = 'Arial'
        mpl.rcParams['font.weight']        = 'regular'
        mpl.rcParams['savefig.format']     = 'pdf'
        mpl.rcParams['savefig.dpi']        = 300
        mpl.rcParams['figure.facecolor']   = 'w'
        mpl.rcParams['figure.edgecolor']   = 'None'
        mpl.rcParams['axes.facecolor']     = 'None'
        mpl.rcParams['legend.frameon']     = False
        mpl.rcParams['axes.spines.right']  = False
        mpl.rcParams['axes.spines.top']    = False
