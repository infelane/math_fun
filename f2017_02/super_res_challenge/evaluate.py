# Evaluation function
# around 20 minutes on GOUGER
# around 3.5 minutes without SSIM


from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from PIL import Image
import numpy as np


def main(test_dir, bool_print = True):

    refdir = '//ipi/scratch/hluong/NTIRE17/DIV2K_train_HR'

    len_dir = 800   # len(glob.glob(refdir + '/*.png'))
    len_dir = 20

    avg_psnr = 0
    avg_ssim = 0

    perf_i = [0, 0]     # PSNR and SSIM

    for index in range(1, len_dir+1):   # assuming gif
        name = '{0:04d}'.format(index)

        file_test = test_dir + '/' + name + 'x2.png'
        file_ref = refdir + '/' + name + '.png'

        im_ref = np.asarray(Image.open(file_ref))
        im_test = np.asarray(Image.open(file_test))

        # COMMENT OUT WHAT NOT INTERESTED IN
        perf_i[0] = psnr(im_ref, im_test)
        perf_i[1] = ssim(im_ref, im_test, multichannel=True)

        if bool_print:
            print("ssim = {p[0]} (should be close to 1) \npsnr = {p[1]} (as high as possible)\n".format(p=perf_i))

        avg_psnr += perf_i[0]
        avg_ssim += perf_i[1]


        print('{:.1f}% Done'.format(index/len_dir*100))


    print('Average SSIM = {}'.format(avg_ssim / len_dir))
    print('Average PSNR = {}'.format(avg_psnr / len_dir))


if __name__ == '__main__':
    print("own method:")
    # test_dir = '//ipi/scratch/hluong/NTIRE17/DIV2K_train_HR_bicubic/X2-interpolation-bicubic'
    test_dir ='/scratch/lameeus/data/challenges/NTIRE17/x2_v1'
    main(test_dir, bool_print = False)
    
    print("bicubic:")
    test_dir = '//ipi/scratch/hluong/NTIRE17/DIV2K_train_HR_bicubic/X2-interpolation-bicubic'
    main(test_dir, bool_print = False)
