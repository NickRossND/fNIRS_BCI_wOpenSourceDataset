This repository was created for the Introduction to Brain Computer Interfaces course. The data located in this repository is open source data from the article listed in the citation below. The code presented here is an adaption from the open access code provided by the research group. The following were changes made to the original code as well as original scripts made for the purposes of conducting neurofeedback studies:

## Changes and Additions:
1. Altered original preprocessing code to fix bugs, remove unnecessary lines of code, and eliminate unwanted plotting functions  
2. Moved from single file to multi-file function based codes for easy debugging and setup
3. Created a 'requirements.txt' file for easy implementation
4. Implemented code for making predictions, both post-hoc and for real-time analysis
5. Introduced measures to use existiing dataset as a means for testing real-time BCI functions 
6. Scripts added to transfer data from python to MATLAB based graphical user interface (GUI) 
7. MATLAB GUI (mlapp) created for conducting neurofeedback studies

## Information and Operation: 
1. The original data collected consisted of EEG-fNIRS recordings released by Weibo Yi, Jiaming Chen, Dan Wang, et al. The only data used in this project and what will be found here is the fNIRS data. The EEG was not used for this work. 
2. Before running this script, please ensure that the dataset was downloaded and saved in the "./Data/FineMI" directory
3. To run the Python program use "fNIRS_decoding_NAR.py". 
4. For operating the MATLAB GUI in simulation mode, open the "GUI.mlapp" and "GUI_communication_test.py" simultaneously. Begin by starting the MATLAB app and inputting 5 seconds into the rest box. Then run the Python file to run a simulation.

## Citation:
If you choose to use any data or code published here, we ask that you please cite the dataset and related paper published on Scientific Data on which this project is based on:
1. Yi, W-B.\*‡, Chen, J-M.‡, Wang, D.\* *figshare* https://doi.org/10.6084/m9.figshare.24123303 (2023).
2. Yi, W.\*‡, Chen, J.‡, Wang, D\*. et al. A multi-modal dataset of electroencephalography and functional near-infrared spectroscopy recordings for motor imagery of multi-types of joints from unilateral upper limb. Sci Data 12, 953 (2025). https://doi.org/10.1038/s41597-025-05286-0


# fNIRS_BCI_wOpenSourceDataset


