import numpy as np

def main():
    lst = ['01000100',
           '01010000',
           '01010011',
           '00101011',
           ]

    lst = lst[:3]
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    lst_int = [int(bin, 2) for  bin in lst]
    
    i_min = min(lst_int)
    i_max = max(lst_int)
    
    print(i_max - i_min, 'should be smaller than 26 to fit in alphabet')
    
    for i in range(len(alphabet) - (i_max - i_min)):
        
        word_lst = [alphabet[foo+i - i_min] for foo in lst_int]
        word = ''.join(word_lst)
        print(word)
    
    print('will probably be dps')
    
    return 1


if __name__ == '__main__':
    main()
