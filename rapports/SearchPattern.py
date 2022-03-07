#find patern in all files, in the curent directory
import os
import time



pattern_not_found = True
user_input =os.getcwd()# name of curent directory
directory = os.listdir(user_input)
searchstring = "rf_optimized"




print('you are in : ', user_input, "\n")
print('directory : ',directory,"\n")




for root, dirs, files in os.walk(user_input):# add ../ to search 1 dir higher
    #print("files :",files,"\n")
    for name in files:
        #print("name file : ",name)
        check = True
        liste={".png",".pdf",".pptx",".docx"}
        for extention in liste:
            if extention in name :
               check = False
        if check:
            full_path=user_input + os.sep + name
            if os.path.isfile(full_path):
                f = open(full_path, 'r')
                if searchstring in f.read():
                    pattern_not_found = False
                    print('found string in file %s' % name)
            f.close()

if(pattern_not_found):
    print("file ",searchstring," not found \n")

print("fonctions declaré a :",time.strftime("%H:%M:%S", time.localtime()))