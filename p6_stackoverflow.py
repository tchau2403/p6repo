#! /usr/bin/env python3
# coding: utf-8

"""
Created on Thu May  3 07:55:00 2018

@author: Thierry CHAUVIER
"""
import time
from datetime import datetime
import argparse
import warnings
import logging as lg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.core.display import display
from IPython.core.display import HTML

import t_lib_util as util


# =============================================================================
#  Load file
# =============================================================================



# =============================================================================
# Main function
# =============================================================================

def main():
    """
    Main function
    """

    print('*'*80)
    print("Starting at ", datetime.now().strftime('%Y_%m_%d_%Hh%Mm%S '))
    deb_prog = time.time()

    #datafile=get_args()
    #df=load_file(datafile)
    df = pd.read_excel("Online Retail.xlsx")

    # Analyze the columns of the file

    print("Starting analyze at ", datetime.now().strftime('%Y_%m_%d_%Hh%Mm%S '))
    #results = util.t_analyze(df)
    #writer = pd.ExcelWriter('Analyse_results.xlsx', engine='xlsxwriter')
    #results.to_excel(writer, sheet_name='Sheet1')
    #writer.save()


    t2d = (time.time()-deb_prog)/60

    print("End of program in %f minutes at "%(t2d), datetime.now().strftime('%Y_%m_%d_%Hh%Mm%S '))
    print('*'*80)
# =============================================================================
#  Start run
# =============================================================================

if __name__ == "__main__":


    # =============================================================================
    # Initialize pandas
    # =============================================================================

    display(HTML("<style>.container { width:100% !important; }</style>"))
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.max_columns', 2000)
    pd.set_option('display.width', 2000)


    # =============================================================================
    # Execute main
    # =============================================================================

    main()
