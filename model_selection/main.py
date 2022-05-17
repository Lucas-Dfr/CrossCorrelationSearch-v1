from model_selection import *


file_names_list = []
for i in range(1, 41):
    if i < 10:
        file_name = 'f00' + str(i) + '_TBabs_diskbb_powerlaw_famodel.xcm'
        file_names_list.append(file_name)
    else :
        file_name = 'f0' + str(i) + '_TBabs_diskbb_powerlaw_famodel.xcm'
        file_names_list.append(file_name)

selected_models = select_models(file_names_list,2.)

for file in selected_models:
    print(file)