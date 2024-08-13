from captcha.image import ImageCaptcha
import random
import os
import string
image = ImageCaptcha(width = 128, height = 32)
char_list = list(string.ascii_letters + string.digits)

def rand_letter(size):
    for i in range(0, size):
        name = ''.join(random.choices(char_list, k=5))
        image.write(name, name+'.png')
        print(f'{round((i/size)*100)}% done', end= '\r')
    


os.mkdir('general_dataset')
os.chdir('general_dataset')
os.mkdir('train')
os.chdir('train')
rand_letter(20000)
os.chdir('..')
os.mkdir('validation')
os.chdir('validation')
rand_letter(5000)
os.chdir('..')