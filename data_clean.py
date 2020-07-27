import os
import glob



#folder = "demo/"
#src_path = folder + "obj_train_data/"

# Rewrite and rename the files.
# 4-->1; 5-->4; 7-->5
#for file_path in glob.iglob(os.path.join(src_path, "*.txt")):
#    name, suffix = os.path.splitext(os.path.basename(file_path))
#    
#    num = int(name.split("_")[1])
#    fout = open(src_path + str(num) + ".txt", "w+")
##        
#    fin = open(file_path, "r+")   
#    for line in fin.readlines():
#        s = ""
#        if line[0] == "4":
#            s = "1" + line[1:]
#        elif line[0] == "5":
#            s = "4" + line[1:] 
#        elif line[0] == "7":
#            s = "5" + line[1:] 
#        else:
#            s = line
#        fout.write(s)
#        
#    fin.close()
#    fout.close()
#    os.remove(file_path)
#    print("Done!")


# Remove useless images / textfile.
#folder = "demo/"
#src_path = folder + "obj_train_data/"
#cnt = 0
#for img_file_path in glob.iglob(os.path.join(src_path, "*.jpg")):
#    name, suffix = os.path.splitext(os.path.basename(img_file_path))
#    if not os.path.exists(src_path + name + ".txt"):
#        os.remove(img_file_path)
#        cnt += 1
#    
#print("Done with " + str(cnt) + "!")

# Reorder all data files.   
#folder = "5-GP010963/"
#src_path = folder + "obj_train_data/"
#dst_path ="data/obj_train_data/"
#
#idx = 13286
#
#for file_path in glob.iglob(os.path.join(src_path, "*.txt")):
#    name, suffix = os.path.splitext(os.path.basename(file_path)) 
#    os.rename(src_path + name + ".txt", dst_path + str(idx) + ".txt")
#    os.rename(src_path + name + ".jpg", dst_path + str(idx) + ".jpg")
#    idx += 1
#
#print("End with idx: " + str(idx))

