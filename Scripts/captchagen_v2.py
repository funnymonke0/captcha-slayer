from captcha.image import ImageCaptcha
import random
import os
import string
image = ImageCaptcha()
char_list = list(string.ascii_letters + string.digits)

def rand_letter(size):
    for i in range(0, size):
        name = ''.join(random.choices(char_list, k=5))
        image.write(name, name+'.png')
        print(f'{round((i/size)*100)}% done', end= '\r')
    
def new_dir(name, entries):
    os.mkdir(name)
    os.chdir(name)
    rand_letter(entries)
    os.chdir('..')

os.mkdir('general_dataset')

new_dir('train', 20000)

new_dir('validation', 5000)

new_dir('test', 2500)

