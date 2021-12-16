# +
# get file size in python
import os
 

def get_all_file(dir_path):
    global files
    for filepath in os.listdir(dir_path):
        tmp_path = os.path.join(dir_path,filepath)
        if os.path.isdir(tmp_path):
            get_all_file(tmp_path)
        else:
            files.append(tmp_path)
    return files

def calc_files_size(files_path):
    files_size = 0
    for f in files_path:
        files_size += os.path.getsize(f)
    return files_size

for sequence in range(1,38):
    path = r'C:/Users/bolinchen3/Desktop/codercompress/arith/txt/latent/bin_binary_golomb/'+str(sequence)+'/'
    ls = os.listdir(path)
    files = []
    files = get_all_file(path)
    overall_bits=calc_files_size(files)*8
    print(overall_bits)
    
    

    
