from captcha.image import ImageCaptcha
import random
import os
import string
image = ImageCaptcha()
choose = list(string.ascii_letters + string.digits)

# def generator(len):
#     string = ''.join(random.choices(choose, k=len))
#     return string

def rand_letter():
    for i in range(0, 62):
        name = choose[i]
        try:
            os.mkdir(name)
            os.chdir(name)
        except FileExistsError:
            os.mkdir('cap'+name)
            os.chdir('cap'+name)
        for j in range (0,500):
            image.write(name, name+'_'+str(j)+'.png')
        os.chdir('..')


os.mkdir('62_dataset')
os.chdir('62_dataset')
os.mkdir('train')
os.chdir('train')
rand_letter()
os.chdir('..')

os.mkdir('validation')
os.chdir('validation')
rand_letter()