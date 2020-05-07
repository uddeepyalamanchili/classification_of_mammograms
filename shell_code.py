import subprocess
list_files = subprocess.run(["python","preproc_san.py","mdb001c.png"],stdout=subprocess.PIPE).stdout.decode('utf-8')
s = subprocess.run(['ls', '-l'], stdout=subprocess.PIPE).stdout.decode('utf-8')
print(list_files.split('\n')[-4])
print(type(s))

