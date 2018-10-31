import numpy as np
import math
import matplotlib.pyplot as plt


def gen_letters(n):
    lst = []
    for i in range(n):
        letter = chr(97+i)
        lst.append(letter)
        
    return lst


def gen_perm(lst):
    
    perms = []
    
    def do_foo(lst_sub, perms, current_string = []):
        
        
    
        for i in range(len(lst_sub)):
            letter_i = lst_sub[i]
            lst_subsub = lst_sub[:i] + lst_sub[i+1:]

            # perms.append([letter_i])

            current_string_copy = list(current_string)

            current_string_copy.append(letter_i)

            do_foo(lst_subsub, perms, current_string_copy)
            
            if len(lst_subsub) == 0:
                if 0:
                    print('depth reached')
                    print(current_string_copy)
                perms.append(current_string_copy)
                
        return perms
        
    return do_foo(lst, perms)

def plot_fac_fac(n_max = 1):
    """
    1, 2, 720, 6.2e23...
    :param n_max:
    :return:
    """
    n = []
    fac_fac = []
    for i in range(1, n_max+1):
        n.append(i)
        fac_fac.append(math.factorial(math.factorial(i)))
        
    print(fac_fac)
        
    plt.figure()
    plt.plot(n, fac_fac)
    plt.show()
    
    
def check_contains_all(prop, perms):
    
    win = 0
    fail = 0
    for perm_i in perms:
        
        string = ''.join(perm_i)
        
        if string in prop:
            win += 1
            
        else:
            print('{} not in it {}'.format(string, prop))
            fail+= 1
            
    print('acc = {}/{} ({}%)'.format(win, win+fail, 100*win/(win+fail)))
    
    
def get_best_lookup(n):
    if n == 1:
        best_string = 'a'   # 1
    elif n == 2:
        best_string = 'aba' # 3
    elif n == 3:
        best_string = 'abcabacba' # 9
    else:
        raise NotImplementedError('no given best string for n = {}'.format(n))
        
    return best_string


def stack_with_overlap(a, b):
    # TODO
    # check overlap
    
    n = len(b)
    for i in range(n-1, 0, -1):
        
        if a[-i:] == b[:i]:
            return a[:-i] + b
        # check if there is an overlap
    
    # there is no overlap
    return a + b


def badly(perms):
    
    n = np.inf
    best_string = None
    
    def foo(rest_prev, n, best_string, current_string = ''):
        for i in range(len(rest_prev)):
            rest = rest_prev[:i] + rest_prev[i+1:]

            next_string = stack_with_overlap(current_string, ''.join(rest_prev[i]))
            
            if len(rest) == 0:
                length = len(next_string)
                print('{} : {}'.format(length, next_string))
                
                if length < n:
                    n = length
                    best_string = next_string
                    
            else:
                n, best_string = foo(rest, n, best_string, next_string)
        return n, best_string

    n, best_string = foo(perms, n, best_string)
    
    print('best ({}) : {}'.format(n, best_string))
    
    return best_string


def v2(perms):
    # includes early stopping
    n = np.inf
    best_string = None
    
    def foo(rest_prev, n, best_string, current_string=''):
        for i in range(len(rest_prev)):
            rest = rest_prev[:i] + rest_prev[i + 1:]
            
            next_string = stack_with_overlap(current_string, ''.join(rest_prev[i]))

            length = len(next_string)
            if length >= n:
                return n, best_string
            
            if len(rest) == 0:
                
                print('{} : {}'.format(length, next_string))
                
                if length < n:
                    n = length
                    best_string = next_string
            
            else:
                n, best_string = foo(rest, n, best_string, next_string)
        return n, best_string
    
    n, best_string = foo(perms, n, best_string)
    
    print('best ({}) : {}'.format(n, best_string))
    
    return best_string


def v3(perms):
    # includes early stopping
    # TODO include suprememum length?
    n = np.inf
    best_string = None
    
    length_single = len(perms[0])
    supremum = 0
    for i in range(1, length_single+1):
        # CITE
        supremum += math.factorial(i)
    
    n = supremum + 1 # +1 to find first one of length 'supremum'
    
    def foo(rest_prev, n, best_string, current_string=''):
        for i in range(len(rest_prev)):
            rest = rest_prev[:i] + rest_prev[i + 1:]
            
            next_string = stack_with_overlap(current_string, ''.join(rest_prev[i]))
            
            length = len(next_string)
            if length >= n:   # if not better: stop!
                return n, best_string
            
            if len(rest) == 0:
                
                print('{} : {}'.format(length, next_string))
                
                if length < n:
                    n = length
                    best_string = next_string
            
            else:
                n, best_string = foo(rest, n, best_string, next_string)
        return n, best_string
    
    n, best_string = foo(perms, n, best_string)
    
    print('best ({}) : {}'.format(n, best_string))
    
    return best_string


def main():
    n = 4
    lst = gen_letters(n)
    
    if 0:
        print('all chars: {}'.format(lst))

    perms = gen_perm(lst)

    print(perms)
    
    if 0:
        plot_fac_fac(4) # this is ridiculously inefficient!
       
    if 0:
        best_lookup = get_best_lookup(n)
        check_contains_all(best_lookup, perms)

    best_gen = v3(perms)

    check_contains_all(best_gen, perms)


if __name__ == '__main__':
    main()