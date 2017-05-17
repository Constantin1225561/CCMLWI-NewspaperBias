# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:35:14 2017

@author: Timo van Niedek
"""

from nytimesarchive import ArchiveAPI
# Note to self: remove API key before pushing to git
api = ArchiveAPI('6faa534a59e44205829aad054e5095ce')

jan = api.query(2017,1)