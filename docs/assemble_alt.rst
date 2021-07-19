Assemble (alt)
==============

The module ``assemble.py`` finds and forms the essential structure components, 
which are the smallest building blocks that form every repeat in the song. These functions 
ensure each time step of a song is contained in at most one of the song's essential structure 
components by ensuring there are no overlapping repeats in time. When repeats overlap, they 
undergo a process where they are divided until there are only non-overlapping pieces left over. 

This module contains the following functions: