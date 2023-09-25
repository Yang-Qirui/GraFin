import numpy as np
import heapq

def dijkstra(adj_matrix, start):
    N = adj_matrix.shape[0] 
    visited = set()
    shortest_distances = {node: float('inf') for node in range(N)}
    shortest_distances[start] = 0
    node_count = {node: 0 for node in range(N)}
    node_count[start] = 1 
    
    priority_queue = [(0, start)]
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        for neighbor in range(N):
            if adj_matrix[current_node, neighbor] > 0:
                distance = current_distance + adj_matrix[current_node, neighbor]
                if distance < shortest_distances[neighbor]:
                    shortest_distances[neighbor] = distance
                    node_count[neighbor] = node_count[current_node] + 1
                    heapq.heappush(priority_queue, (distance, neighbor))
                    
    return np.array(list(shortest_distances.values())), np.array(list(node_count.values()))

def gen_landmarks_features(adjacent_matrix, thres_hop, landmark_num):
    '''
        first, find the shortest path based on adjacent matrix
        then, use threshold to filter
        return the features of all APs
    '''
    if thres_hop > 1:
        shortest_distances = []
        for i in range(landmark_num):
            shortest_dist, node_count = dijkstra(adjacent_matrix, i)
            mask = node_count > thres_hop + 1
            shortest_dist[mask] = 0
            shortest_distances.append(shortest_dist)
        landmark_features = np.array(shortest_distances)[:, landmark_num:]
        assert landmark_features.shape == (landmark_num, adjacent_matrix.shape[1] - landmark_num)
        return landmark_features.T
    elif thres_hop == 1:
        return adjacent_matrix[:landmark_num].T # equals to adjacent matrix