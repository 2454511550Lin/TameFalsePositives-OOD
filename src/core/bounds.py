import logging 
import math
# define different upper confidence bounds

from math import log, sqrt

def lil_theory(t,delta,beta_t,p,lambda_max,lambda_min,eta=0.001):
    ct = 1-beta_t+beta_t/(p**2)
    lambda_dis = lambda_max - lambda_min
    if lambda_dis < 10:
        lambda_dis = 10
    try:
        result = sqrt( (3*ct/t) * (2*log(log(3*ct*t/2)) + 
                                   log((2/delta)*log((lambda_dis)/eta)) ) )
        return result
    except ValueError:
        print("t:{} delta:{} beta_t:{} p:{} lambda_max:{} lambda_min:{} eta:{}".format(t,delta,beta_t,p,lambda_max,lambda_min,eta))
        exit(-27)

def lil_heuristic(t, delta ,beta_t,p,c1=0.5,c2=4.75,c3 = 1):
    ct = 1-beta_t+beta_t/(p**2)
    try:
        result = c1* sqrt( (ct/t) * (log(log(c2*ct*t)) + log(c3/delta) ) )
        return result
    except ValueError:
        logging.ERROR("LIL Bound math domain error! t:{} delta:{} c1:{} c2:{} ct:{}".format(t, delta, c1, c2 ,ct))
        exit(-27)

def hoeffding(t, delta):
    return sqrt((1/t)*log(1/delta))
