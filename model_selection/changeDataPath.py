import os 

""" 
This is a script to change the name of the data path in xcm files from 
/Users/db_xifu2/Documents//NICER_DATA/1820-303/BURST_SPECTRA/burst_0199sig_num1_1050300109_intervals_for1000counts/grouping_step1_bin1/b_XXX_opt.pha
to b_XXX_opt.pha in order to remove hard coded path
"""

HEAD = '/Users/lucas/Documents/IRAP/CrossCorrelationSearch/data/BURST_SPECTRA/burst_0217sig_num1_2050300110_intervals_for1000counts/grouping_step1_bin1'

for i in range(1, 54):
    if i < 10:
        file_name = 'f00' + str(i) + '_TBabs_diskbb_powerlaw_famodel.xcm'
        data_file = 'b_00' + str(i) + '_opt.pha'
    else :
        file_name = 'f0' + str(i) + '_TBabs_diskbb_powerlaw_famodel.xcm'
        data_file = 'b_0' + str(i) + '_opt.pha'
    
    path = os.path.join(HEAD, file_name)
    
    # Open file and get the list of lines
    f = open(path, 'r')
    list_of_lines = f.readlines()
    # Now change line 3 to specify new path
    list_of_lines[3] = 'data 1:1 ' + data_file + "\n"
    
    # Imprint changes on the file
    f = open(path, 'w')
    f.writelines(list_of_lines)
    f.close()
    
    
    
    
    
        
    