import numpy as np
import matplotlib.pyplot as plt

map = [[13326, 5568], [9532, 1370], [3610, 4421], [7994, 7900]]
map = [[13326, 5568], [3610, 4421], [9532, 1370], [7994, 7900]] # shuffled
tot_laps = 3

info_values = []


class State():
    def __init__(self):
        self.i_laps_done = 0
        

def get_dif_node(i, dif=0):
    i_new = (i + dif) % len(map)
    
    return map[i_new]

class InfoNow(object):
    def __init__(self, cox, coy, vx, vy, angle, i_node):
        self.co = np.array([cox, coy])
        self.v =  np.array([vx, vy])
        self.angle = angle
        self.i_node = i_node
        

def angle_from_co_s(co1, co2):
            dif = [b - a for a, b in zip(co1, co2)]
            angle = (np.arctan2(dif[1], dif[0]) * 180 / np.pi) % 360    # y, x!
            return angle


def angle_to_vec(angle):
    rad = angle * np.pi / 180
    return np.array([np.cos(rad), np.sin(rad)])


def vec_to_angle(vec):
    angle = (np.arctan2(vec[1], vec[0]) * 180 / np.pi) % 360  # y, x!
    return angle


def dif_angles(angle1, angle2):
    return (angle2 - angle1 + 180) % 360 - 180


def pred_next_step(info_now, output, map, state):
    # DO only for first one now
    
    def angle_to_vec(angle):
        rad = angle * np.pi / 180
        return np.array([np.cos(rad), np.sin(rad)])

    """
    prediction:
    """

    co_now = info_now.co
    v_vec_cur = info_now.v
    angle_current = info_now.angle
    co_target = output[0:2]
    thrust = output[2]

    angle_next = angle_from_co_s(co_now, co_target)
    #
    # print_error(angle_next)
    # print_error(angle_current)

    # next step:
    # 1 rotate:
    if angle_current == -1:
        angle_pred = angle_next
    else:
        dif = (angle_next - angle_current + 180) % 360 - 180
        angle_pred = angle_current + max(-18, min(+18, dif))

    # 2. acceleration
    facing_vector = angle_to_vec(angle_pred)
    if thrust == 'BOOST':
        thrust_vector = 650 * facing_vector
    else:
        thrust_vector = thrust * facing_vector


    v_vec_pred = v_vec_cur + thrust_vector

    # 3 movement
    co_pred = co_now + v_vec_pred

    # TODO incorporate collision

    # 4 friction
    v_vec_pred = 0.85 * v_vec_pred

    # 5. correcting modulo
    angle_pred = int(np.round(angle_pred % 360))
    co_pred = (np.round(co_pred)).astype(int)
    v_vec_pred = (v_vec_pred).astype(int)  # truncated

    co_node = get_dif_node(info_now.i_node)
    dist_map = np.linalg.norm(co_node - co_pred)
    if dist_map < 600:
        # check if finish reached
        if info_now.i_node == 0:
            state.i_laps_done += 1
            
        i_node = (info_now.i_node + 1) % len(map)

    else:
        i_node = info_now.i_node

    info_next = InfoNow(*co_pred, *v_vec_pred, angle_pred, i_node)
    return info_next

class Agent():
    def __init__(self, a = None, b = None, update_from=None):
        if a is None:
            self.a = np.random.normal()
        else:
            self.a = a
        if b is None:
            self.b = np.random.normal()
        else:
            self.b = b
            
        if update_from is None:
            self.kernel = np.random.normal(size=(1, 1))
            self.bias = np.random.normal(size=(1, 1))
        else:
            self.kernel = update_from.kernel + np.random.normal(size=(1, 1))
            self.bias = update_from.bias + np.random.normal(size=(1, 1))
        
    def next_move(self, info_now):
        # basic data:
        co_now = info_now.co
        co_node = get_dif_node(info_now.i_node)
        v_now = info_now.v
    
        # possible usefull data:
        dist_node = np.linalg.norm(co_node - co_now)
    
        # calculations
    
        co_target = get_dif_node(info_now.i_node)
    
        if dist_node < self.a:
            # slow down before point?
            thrust = 0
        else:
            thrust = 100

        dist_node_norm = (dist_node - 6000.)/3000.

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        # print(dist_node_norm * self.a + self.b)
        # print(sigmoid(dist_node_norm * self.a + self.b))
        
        thrust = 100.*sigmoid(dist_node_norm * self.a + self.b)
        
        angle_node = angle_from_co_s(co_now, co_node)
        angle_v = vec_to_angle(v_now)
        
        d_angle_speed_node = dif_angles(angle_v, angle_node)
        
        info_values.append(d_angle_speed_node)

        d_angle_target = d_angle_speed_node*(1+self.kernel[0, 0]) + self.bias[0, 0]
        
        angle_target = angle_v + d_angle_target
        
        vec_target = 1000*angle_to_vec(angle_target)    # just make sure it is large enough
        co_target = co_now + vec_target
        # print(thrust)
    
        return [*co_target, thrust]
        

# def agent(info_now, settings = {'a': 0}):
#     # basic data:
#     co_now = info_now.co
#     co_node = get_dif_node(info_now.i_node)
#
#     # possible usefull data:
#     dist_node = np.linalg.norm(co_node - co_now)
#
#     # calculations
#
#     co_target = get_dif_node(info_now.i_node)
#
#     if dist_node < settings['x']:
#         # slow down before point?
#         thrust = 0
#     else:
#         thrust = 100
#
#     return [*co_target, thrust]

def optimal_output():
    """
    Originally you have the coordinates of ship, next node and one before, angle of ship and speed vector
    
    rewrite to:
        distance to node
        facing angle
        speed vector
        angle (ship, node, next)
        distance (node, next)
    """
    
def main():
    t_max = 500
    # co_start = pod_start[0:2]
    
    if 0:
        # evolution
        n_iters = 10
        n_agents = 10
        n_keep = 5
        
        agents = [Agent() for _ in range(n_agents)]
        
    else:
        n_iters = 10
        n_agents = 10
        n_keep = 5
        agents = [Agent(0, 1000) for _ in range(n_agents)]
    
    def foo(agent_i, with_plot = False):
        state = State()
        pod_start = [*map[0], 0, 0, -1, 1]  # v starts at 0, angle unknown
        
        info_now = InfoNow(*pod_start)

        co_list = []
        v_list = []
        angle_list = []
        
        t = 0
        while t < t_max and state.i_laps_done < 3:
            output = agent_i.next_move(info_now)
        
            info_now = pred_next_step(info_now, output, map, state)
        
            co_list.append(info_now.co)
            v_list.append(info_now.v)
            angle_list.append(info_now.angle)
        
            #
            # if info_now.co[0] < 0:
            #     co_target = co_start   # turn arround
        
            t += 1
    
        print('finished after: {}'.format(t))
        
        if with_plot:
        
            x_list, y_list = zip(*co_list)
        
            ax = plt.axes()
        
            for i in range(len(co_list)):
                ci = co_list[i]
            
                v_i = v_list[i]
                angle_i = angle_list[i]
                aim_i = 300 * angle_to_vec(angle_i)
            
                if 0:
                    ax.arrow(*ci, *v_i, head_width=100, head_length=100, fc='r', ec='r')
                    ax.arrow(*ci, *aim_i, head_width=100, head_length=100, fc='b', ec='b')
        
            map_x, map_y = zip(*map)
            plt.plot(map_x, map_y, 'x', markersize=20)
        
            plt.plot(x_list, y_list)
            plt.show()
        
        return t
        
    t_best = []
    t_aver = []
    
    for iter in range(n_iters):
        
        costs = []
        
        for agent_i in agents:
            t_i = foo(agent_i)
            costs.append(t_i)

        idx = np.argsort(costs)

        t_best.append(costs[idx[0]])

        agents_sorted = [agents[i] for i in idx]
        costs_sorted = [costs[i] for i in idx]
        
        best_agents = agents_sorted[:n_keep]
        best_costs = costs_sorted[:n_keep]
        
        t_aver.append(np.mean(best_costs))

        new_agents = []
        for i in range(n_agents - n_keep):
            i_random = np.random.randint(n_keep)
            good_agent = best_agents[i_random]
            agent_new_i = Agent(a=good_agent.a + np.random.normal(), b=good_agent.b + np.random.normal(), update_from=good_agent)
            new_agents.append(agent_new_i)
        
        agents = best_agents + new_agents
    
    """ some extra info """
    print('{} and {}'.format(np.mean(info_values), np.std(info_values)))
    
    plt.figure()
    plt.plot(t_best)
    plt.plot(t_aver)
    plt.figure()
    
    best_agent = agents_sorted[0]

    foo(best_agent, with_plot=True)
    
    
if __name__ == '__main__':
    main()