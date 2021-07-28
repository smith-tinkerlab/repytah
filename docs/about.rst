Why repytah
===========

Sequential data streams often have repeated elements that build on each other, creating hierarchies. 
Therefore, the goal of the Python package ``repytah`` is to extract these repetitions and their relationships 
to each other in order to form aligned hierarchies, a low-dimensional representation for sequential data
that synthesizes and aligns all possible hierarchies of the meaningful repetitive structure in a sequential data stream 
along a common time axis. 

An example of sequential data streams is music-based data streams, such as songs. In addition to 
visualizing the structure of a song, aligned hierarchies can be embedded into a classification space with a 
natural notion of distance and post-processed to narrow the exploration of a music-based data stream to certain 
lengths of structure, or to address numerous MIR tasks, including the cover song task, the segmentation task, 
and the chorus detection task. 

For future work, based on the aligned hierarchies, we can build aligned sub-hierarchies. Aligned sub-hierarchies 
are a collection of individual aligned hierarchies for each repeated pattern in a given sequential
data. With aligned sub-hierarchies, we can better deal with tasks that require a degree of flexibility. For
example, we can implement the aligned sub-hierarchies to find all versions of the same piece of music based
on a given version of the recording, which might involve different expressions including the number of
repeats.
