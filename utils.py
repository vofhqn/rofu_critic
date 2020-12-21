import math
import torch
import copy
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def rollout(state, env, policy, length):
    env.state = state 
    results = []
    for l in range(length):
        s = copy.deepcopy(env.state)
        a = policy.select_action(s)
        ns, r, done, _ = env.step(a)
        results.append((s, a, ns, r, done))
    return results

def train_model(fakeenv, memory, batch_size, threshold=10, log=False):
    cur_mse = 1e8
    count = 0
    iter_count = 0
    while count < threshold:
        mse = fakeenv.train(memory, batch_size).cpu().data.numpy()
        #print("mse", mse)
        if cur_mse - mse < 0.002:
            count += 1
        iter_count += 1
        cur_mse = mse
    if log:
        print("model error: ", cur_mse, "iter_count", iter_count)

def terminate_fn(_obs, _act, _next_obs, env_name):
    if type(_obs) is torch.Tensor:
        obs = _obs.cpu().data.numpy()
        act = _act.cpu().data.numpy()
        next_obs = _next_obs.cpu().data.numpy()
    else:
        obs = _obs
        act = _act
        next_obs = _next_obs

    if env_name == "Ant-v2":
        x = next_obs[:, 0]
        not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * (x >= 0.2) \
                    * (x <= 1.0)

        done = ~not_done
        done = done[:,None]
        return done
    if env_name == "Halfcheetah-v2":
        done = np.array([False]).repeat(len(obs))
        done = done[:,None]
        return done
    if env_name == "Hopper-v2":
        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:,None]
        return done

    if env_name == "Humanoid-v2":
        z = next_obs[:,0]
        done = (z < 1.0) + (z > 2.0)

        done = done[:,None]
        return done

    if env_name == "Walker2d-v2":
        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done =  (height > 0.8) \
                    * (height < 2.0) \
                    * (angle > -1.0) \
                    * (angle < 1.0)
        done = ~not_done
        done = done[:,None]
        return done

    if env_name == "InvertedPendulum-v2":

        notdone = np.isfinite(next_obs).all(axis=-1) \
                  * (np.abs(next_obs[:,1]) <= .2)
        done = ~notdone

        done = done[:,None]

        #print("oo?", obs, act, next_obs, env_name, done)
        return done
    return np.array([False]).repeat(len(obs))
    raise NotImplemented

def dim_resize(state, diag, shift):
    b, d = state.shape

    mat1 = torch.diag(diag).to(device)
    _state = torch.matmul(state, mat1)
    __state = _state + shift

    return __state

def resize(state, env_name):
    if env_name == "Walker2d-v2":
        #return state
        b, d = state.shape
        diag = 10.* torch.ones(d).to(device)
        diag[0] = 3.
        diag[1] = 2.
        # diag[-1] = 20.#reward
        shift = torch.zeros(b,d).to(device)
        #shift[:, 0] = 1.4
        state1 = dim_resize(state, diag, shift)
        return state1