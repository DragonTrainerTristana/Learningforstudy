import numpy as np
import matplotlib.pyplot as plt

sigma_r = 1.0
epsilon_M = 1.0
lambda_coeff = 0.5

# Define 4x4 grid nodes
nodes = {
    0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0),
    4: (0, 1), 5: (1, 1), 6: (2, 1), 7: (3, 1),
    8: (0, 2), 9: (1, 2), 10: (2, 2), 11: (3, 2),
    12: (0, 3), 13: (1, 3), 14: (2, 3), 15: (3, 3),
}

# Define edges for 4x4 grid
edges = {}
for i in range(4):
    for j in range(4):
        node = i * 4 + j
        if j < 3:  # Horizontal edge
            edges[(node, node + 1)] = 1.0
        if i < 3:  # Vertical edge
            edges[(node, node + 4)] = 1.0

# Initialize robots
robots = [
    {"id": 0, "pos": 0, "visited": set(), "messages": {}, "path": []},
    {"id": 1, "pos": 15, "visited": set(), "messages": {}, "path": []},
]

def euclidean_distance(node_a, node_b):
    if isinstance(node_a, int):
        x1, y1 = nodes[node_a]
    else:
        x1, y1 = node_a
    x2, y2 = nodes[node_b]
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def generate_message(robot, target_node, other_robot_positions, overlap_count):
    dir_robots = [euclidean_distance(target_node, pos) for pos in other_robot_positions]
    repulsion = -sigma_r * sum(d**2 for d in dir_robots)
    penalty = -epsilon_M * overlap_count
    message = repulsion + penalty
    return message

def assign_zones(nodes, robots):
    zone_size = len(nodes) // len(robots)
    for i, robot in enumerate(robots):
        robot["zone"] = set(range(i * zone_size, (i + 1) * zone_size))

def select_next_node(robot):
    best_node = None
    best_priority = float('-inf')
    for target_node in nodes:
        if target_node in robot["visited"]:
            continue
        distance_cost = edges.get((robot["pos"], target_node), edges.get((target_node, robot["pos"]), float('inf')))
        zone_bonus = 10 if target_node in robot["zone"] else 0  # 우선적으로 자기 영역을 탐색
        overlap_count = sum(1 for r in robots if target_node in r["visited"])
        repulsion = -sigma_r * overlap_count
        priority = -distance_cost + repulsion + zone_bonus
        if priority > best_priority:
            best_priority = priority
            best_node = target_node
    return best_node

def exploration_loop(robots, max_iterations=50):
    for _ in range(max_iterations):
        exchange_messages(robots)
        for robot in robots:
            next_node = select_next_node(robot)
            if next_node is not None:
                robot["visited"].add(next_node)
                robot["path"].append((robot["pos"], next_node))
                robot["pos"] = next_node
        all_visited = set(node for r in robots for node in r["visited"])
        if len(all_visited) == len(nodes):
            break

def visualize_grid(nodes, robots, grid_size=(4, 4)):
    plt.figure(figsize=(6, 6))
    for x in range(grid_size[0] + 1):
        plt.axvline(x, color="gray", linestyle="--", linewidth=0.5)
    for y in range(grid_size[1] + 1):
        plt.axhline(y, color="gray", linestyle="--", linewidth=0.5)

    colors = ["blue", "orange"]
    for i, robot in enumerate(robots):
        for (start, end) in robot["path"]:
            x1, y1 = nodes[start]
            x2, y2 = nodes[end]
            plt.plot([x1, x2], [y1, y2], marker="o", color=colors[i], label=f"Robot {robot['id']} path" if (start, end) == robot["path"][0] else "")
        for visited in robot["visited"]:
            x, y = nodes[visited]
            plt.scatter(x, y, color=colors[i], s=100)

    for node, (x, y) in nodes.items():
        plt.scatter(x, y, color="black", s=200, zorder=5)
        plt.text(x, y, f" {node}", fontsize=12, zorder=10)

    plt.xlim(-0.5, grid_size[0] - 0.5)
    plt.ylim(-0.5, grid_size[1] - 0.5)
    plt.xticks(range(grid_size[0]))
    plt.yticks(range(grid_size[1]))
    plt.title("Robot Exploration on 4x4 Grid")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(color="gray", linestyle="--", linewidth=0.5)
    plt.show()

assign_zones(nodes, robots)  # 탐색 영역 분배
exploration_loop(robots)
visualize_grid(nodes, robots, grid_size=(4, 4))
