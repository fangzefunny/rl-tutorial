import numpy as np 
from scipy.special import softmax 

from utils.env import frozen_lake

rng = np.random.RandomState(1234)

def policy_eval(pi, env, theta=1e-4, gamma=.99):

    # initialize V(s), arbitrarily except V(terminal)=0
    V = rng.rand(env.nS) * 0.001
    # except v(terminal) = 0
    for s in env.s_termination:
        V[s] = 0

    # loop until convergence
    while True: 
        delta = 0
        for s in env.S:
            v_old = V[s].copy()
            v_new = 0
            for a in env.A:
                p = env.p_s_next(s, a)
                for s_next in env.S:
                    r, done = env.r(s_next)
                    v_new += pi[s, a]*p[s_next]*(r + (1-done)*gamma*V[s_next])
            V[s] = v_new 
            # check convergence
            delta = np.max([delta, np.abs(v_new - v_old)])
        
        if delta < theta:
            break 
    
    return V


def policy_eval2(pi, env, theta=1e-4, gamma=.99):

    p_trans_mat = np.zeros([env.nS, env.nA, env.nS])
    for s in env.S:
        for a in env.A:
            p_trans_mat[s, a, :] = env.p_s_next(s, a)
    R_mat = np.array([env.r(s)[0] for s in env.S]).reshape([-1, 1])
    done_mat = np.array([env.r(s)[1] for s in env.S]).reshape([-1, 1])

     # initialize V(s), arbitrarily except V(terminal)=0
    V = rng.rand(env.nS).reshape([-1, 1]) * 0.001
    
    while True:
        
        v_old = V.copy()
        q = np.squeeze(p_trans_mat@(R_mat + (1-done_mat)*gamma*V))
        V = (pi*q).sum(1, keepdims=True)
        delta = np.abs(V - v_old)
        if (delta < theta).all():
            break 

    for s in env.s_termination:
        V[s] = 0

    return V.reshape([-1])

def policy_iter2(env, seed=1234):

    # matrify the variable 
    p_trans_mat = np.zeros([env.nS, env.nA, env.nS])
    for s in env.S:
        for a in env.A:
            p_trans_mat[s, a, :] = env.p_s_next(s, a)
    R_mat = np.array([env.r(s)[0] for s in env.S]).reshape([-1, 1])
    done_mat = np.array([env.r(s)[1] for s in env.S]).reshape([-1, 1])

    def policy_eval2(pi, V, env, theta=1e-4, gamma=.99):

        # initialize V(s), arbitrarily except V(terminal)=0
        V = rng.rand(env.nS).reshape([-1, 1]) * 0.001
        
        while True:
            
            v_old = V.copy()
            q = np.squeeze(p_trans_mat@(R_mat + (1-done_mat)*gamma*V))
            V = (pi*q).sum(1, keepdims=True)
            delta = np.abs(V - v_old)
            if (delta < theta).all():
                break 

        for s in env.s_termination:
            V[s] = 0

        return V
    
    def policy_improve2(pi, V, env, theta=1e-4, gamma=.99):
        pi_old = pi.copy()
        q = np.squeeze(p_trans_mat@(R_mat + (1-done_mat)*gamma*V))
        pi = np.eye(env.nA)[np.argmax(q, axis=1)]
        # check stable
        stable = (np.abs(pi - pi_old) < theta).all()

        return pi, stable  
    
    rng = np.random.RandomState(seed)

    # initialize V(s), arbitrarily except V(terminal)=0
    V = rng.rand(env.nS).reshape([-1, 1]) * 0.001
    # except v(terminal) = 0
    for s in env.s_termination:
        V[s] = 0
    # initialize π(s), arbitrarily
    pi = softmax(rng.rand(env.nS, env.nA)*5, axis=1)

    while True: 
        V = policy_eval2(pi, V, env)
        pi, stable = policy_improve2(pi, V, env)
        if stable: break

    return V.reshape([-1]), pi 


if __name__ == '__main__':

    env = frozen_lake()
    policy_iter2(env)