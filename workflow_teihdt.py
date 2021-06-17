#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annotation Workflow

- converts TEI master files to annotated TEI files
- annotation with HeidelTime

The final results are stored in a folder "teia".
Run this file directly.

OBS: HeidelTime uses TreeTagger for the linguistic annotation of the corpus files. Before using this workflow,
the path to the TreeTagger installation needs to be set in the file config.props in the HeidelTime directory.
Here the standalone version of HeidelTime is used.


@author: Ulrike Henny-Krahmer
@filename: workflow_teihdt.py 

"""

############ Options ##############

# where the TEI master files are
infolder = "/home/ulrike/Git/conha19/tei"

# where the annotation working files and results should go
outfolder = "/home/ulrike/Git/conha19/heideltime"

# language of the texts (tested for: FRENCH, SPANISH, ITALIAN, PORTUGUESE)
lang = "SPANISH"

# path to heideltime installation
heideltimePath = "/home/ulrike/Programme/heideltime-standalone-2.2.1"


import sys
import os

# use the following to add a path to syspath (if needed):
#sys.path.append(os.path.abspath("/home/ulrike/Git/"))

import prepare_tei
import use_heideltime


# by default, it should be enough to change the options above and leave this as is

#prepare_tei.prepare("split-1", infolder, outfolder)
#use_heideltime.apply_ht(heideltimePath, os.path.join(outfolder, "txt"), os.path.join(outfolder, "hdt"), lang)
#use_heideltime.debug_ampersands(os.path.join(outfolder, "hdt"), os.path.join(outfolder, "anno_pre"))
#use_heideltime.wrap_body(os.path.join(outfolder, "anno_pre"), os.path.join(outfolder, "annotated_temp"))
#prepare_tei.prepare("merge-hdt", outfolder, os.path.join(outfolder, "teia"))
