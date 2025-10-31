# Stony Brook 2025 Summer REU
###### Colton Addis

This work is the final code done for the Stony Brook 2025 Summer REU, with some intermediate and practice files in the mix. Most code in the folders is done for experimenting and/or incomplete versions that are here for completeness, though the code in the binning section is still very useful and relevant.

`unfolding.py` and `utility.py` conatin most of the code needed to implement unfolding on your own, and `mc_data_analysis.ipynb` is the final code used for the poster. Note that it is not quite an example of how to use the functions in the other packages, as some things were moderlized afterwards for ease of implenation later. 

Using `unfolding.py` is mainly done by calling `RLUnfolding2D` with the required information. The file also includes ways to store A-Matrices and to retreive them, to make it easy to mix-and-match matrices. As currently inplemented, the creation of the A-Matrix is left to the user. 

In `mc_data_anaylsis.ipynb`, there is a couple of tuning variables. In the second code block, the file and data and be put it, and the correct weighting timespan can be used. In the third block, the range is the energy range desired. The recon bin and true bin, for energy and zenith respectively, is the number of bins desired. This can either be a list of edges or a number.  
