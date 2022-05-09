import random

def RaTe(num,rate):
    t = int(num * rate)
    R = []
    while(1):
        for i in range(num):
            if t == 0:
                R.append(0)
            else:
                r = random.random()
                if r <= rate:
                    R.append(1)
                    t -= 1
                else:
                    R.append(0)
        if t == 0:
            break
        else:
            continue
    return R
