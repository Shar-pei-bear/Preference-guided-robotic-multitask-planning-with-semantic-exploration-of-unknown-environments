from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
def upper_bound(probs, gains, dfa_state):
    graph = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            if (j <= i):
                continue

            diff = j - i
            one_bit_diff =  diff == (diff & -diff)
            if not one_bit_diff:
                continue

            diff_bit = np.log2(diff).astype(np.uint8)
            if (1 << diff_bit) & i:
                continue
            if probs[diff_bit] <= 0:
                continue

            graph[i, j] = -np.log(probs[diff_bit])

            if np.isclose(graph[i, j], 0):
                graph[i, j] = -1

    graph = csr_matrix(graph)
    graph[graph == -1] = 0
    dist_matrix = dijkstra(csgraph=graph, directed=True, indices=dfa_state)
    dist_matrix = np.exp(-dist_matrix)
    # print(dist_matrix)
    # print(dist_matrix* gains)
    return np.amax(dist_matrix* gains)


def convertToTernary(N):
    # Base case
    # Finding the remainder
    # when N is divided by 3
    s = []
    while N:
        x = N % 3
        N //= 3
        s.append(x)

    while len(s) < 5:
        s.append(0)

    return s[::-1]


def convertToDecimal(N):
    n = 0
    for i in N:
        n = 3*n + i

    return n

def isValid(np_shape, index):
    index = np.array(index)
    return (index >= 0).all() and (index < np_shape).all()


def upper_bound_new(probs, gains, current_dfa_state, current_state, dfa_states, dfa_states_indices, grid_map, grip_map, goal_map, action_deltas):
    classes = ['train', 'car', 'stegosaurus', 'parasaurolophus', 'tank']
    state_space_grid = (grid_map == 255).astype(np.uint8)
    states = np.nonzero(state_space_grid)
    states_index = np.ravel_multi_index(states, grid_map.shape)
    states = np.asarray(states).T

    states_indices = {}
    for i in range(len(states_index)):
        states_indices[states_index[i]] = i

    initial_state = dfa_states_indices[convertToDecimal(current_dfa_state)] * len(states) + states_indices[np.ravel_multi_index(current_state, grid_map.shape)]

    row = []
    col = []
    data = []

    goal_map_states = np.nonzero(goal_map)
    goal_map_states = np.ravel_multi_index(goal_map_states, grid_map.shape)

    for j in range(len(dfa_states)):
        dfa_state = dfa_states[j]
        dfa_state_Ternary = convertToTernary(dfa_state)
        for state in goal_map_states:
            row_index = j * len(states) + states_indices[state]
            for k in range(5):
                if dfa_state_Ternary[k] == 1:
                    data.append(0)
                    next_dfa_state = list(dfa_state_Ternary)
                    next_dfa_state[k] = 2

                    col_index = dfa_states_indices[convertToDecimal(next_dfa_state)] * len(states) + states_indices[state]
                    row.append(row_index)
                    col.append(col_index)
                    break

    grip_maps_states = []
    for k in range(5):
        grip_map_states = np.nonzero(grip_map[classes[k]])
        grip_maps_states.append(np.ravel_multi_index(grip_map_states, grid_map.shape))

    for j in range(len(dfa_states)):
        dfa_state = dfa_states[j]
        dfa_state_Ternary = convertToTernary(dfa_state)

        if dfa_state_Ternary.count(1):
            continue

        for k in range(5):
            if probs[k] <= 0:
                continue

            if dfa_state_Ternary[k]:
                continue

            for state in grip_maps_states[k]:
                row_index = j * len(states) + states_indices[state]
                data.append(-np.log(probs[k]))
                next_dfa_state = list(dfa_state_Ternary)
                next_dfa_state[k] = 1

                col_index = dfa_states_indices[convertToDecimal(next_dfa_state)] * len(states) + states_indices[state]
                row.append(row_index)
                col.append(col_index)



    for i in range(len(states)):
        state = states[i]
        for action_delta in action_deltas:
            next_state = state + action_delta
            if not isValid(grid_map.shape, next_state):
                continue

            if grid_map[tuple(next_state)] != 255:
                continue
            next_state_index = np.ravel_multi_index(next_state, grid_map.shape)
            for j in range(len(dfa_states)):
                row_index = j * len(states) + i
                data.append(0)
                col_index = j * len(states) + states_indices[next_state_index]
                row.append(row_index)
                col.append(col_index)

    print('construct sparse graph')
    graph = csr_matrix((data, (row, col)), shape=(len(dfa_states)*len(states), len(dfa_states)*len(states)))
    print('solving dijkstra algorithm')
    dist_matrix = dijkstra(csgraph=graph, directed=True, indices=initial_state)
    dist_matrix = np.exp(-dist_matrix)
    dist_matrix = np.reshape(dist_matrix, (len(dfa_states), len(states))).T

    return np.amax(dist_matrix * gains)

def upper_bound_efficient(probs, gains, current_dfa_state, current_state, dfa_states, dfa_states_indices, grid_map, grip_map, goal_map):
    state_space_grid = (grid_map == 255).astype(np.uint8)
    states = np.nonzero(state_space_grid)
    states_index = np.ravel_multi_index(states, grid_map.shape)

    classes = ['train', 'car', 'stegosaurus', 'parasaurolophus', 'tank']
    grip_maps_states = []
    for k in range(5):
        grip_map_states = np.nonzero(grip_map[classes[k]])
        grip_maps_states.append(np.ravel_multi_index(grip_map_states, grid_map.shape))

    goal_map_states = np.nonzero(goal_map)
    goal_map_states = np.ravel_multi_index(goal_map_states, grid_map.shape)

    states = np.asarray(states).T
    states_indices = {}
    for i in range(len(states_index)):
        states_indices[states_index[i]] = i

    current_state_index = states_indices[np.ravel_multi_index(current_state, grid_map.shape)]
    initial_state = dfa_states_indices[convertToDecimal(current_dfa_state)] * len(states) + current_state_index

    row = []
    col = []
    data = []

    for j in range(len(dfa_states)):
        dfa_state = dfa_states[j]
        dfa_state_Ternary = convertToTernary(dfa_state)

        if dfa_state_Ternary.count(1) != 1:
            continue

        k = dfa_state_Ternary.index(1)

        contracted_state_index = j * len(states) + current_state_index
        for state in goal_map_states:
            row_index = j * len(states) + states_indices[state]
            if dfa_state_Ternary[k] == 1:
                data.append(0)
                next_dfa_state = list(dfa_state_Ternary)
                next_dfa_state[k] = 2

                col_index = dfa_states_indices[convertToDecimal(next_dfa_state)] * len(states) + states_indices[state]
                row.append(row_index)
                col.append(col_index)

                if contracted_state_index != row_index:
                    data.append(0)
                    row.append(contracted_state_index)
                    col.append(row_index)

                contracted_next_state_index = dfa_states_indices[convertToDecimal(next_dfa_state)] * len(states) + current_state_index
                if contracted_next_state_index != col_index:
                    data.append(0)
                    row.append(col_index)
                    col.append(contracted_next_state_index)


    for j in range(len(dfa_states)):
        dfa_state = dfa_states[j]
        dfa_state_Ternary = convertToTernary(dfa_state)
        if dfa_state_Ternary.count(1):
            continue

        contracted_state_index = j * len(states) + current_state_index

        for k in range(5):
            if probs[k] <= 0:
                continue

            if dfa_state_Ternary[k]:
                continue

            for state in grip_maps_states[k]:
                row_index = j * len(states) + states_indices[state]
                data.append(-np.log(probs[k]))

                next_dfa_state = list(dfa_state_Ternary)
                next_dfa_state[k] = 1

                col_index = dfa_states_indices[convertToDecimal(next_dfa_state)] * len(states) + states_indices[state]
                row.append(row_index)
                col.append(col_index)

                if contracted_state_index != row_index:
                    data.append(0)
                    row.append(contracted_state_index)
                    col.append(row_index)

                contracted_next_state_index = dfa_states_indices[convertToDecimal(next_dfa_state)] * len(states) + current_state_index

                if contracted_next_state_index != col_index:
                    data.append(0)
                    row.append(col_index)
                    col.append(contracted_next_state_index)


    graph = csr_matrix((data, (row, col)), shape=(len(dfa_states)*len(states), len(dfa_states)*len(states)))
    dist_matrix = dijkstra(csgraph=graph, directed=True, indices=initial_state)
    dist_matrix = np.exp(-dist_matrix)
    dist_matrix = np.reshape(dist_matrix, (len(dfa_states), -1)).T

    return np.amax(dist_matrix * gains)


 