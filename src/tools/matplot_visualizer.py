import matplotlib.pyplot as plt
import heapq


class MatplotVisualizer:

    @staticmethod
    def plot_plan(route):
        x = [wp.transform.location.x for wp in route]
        y = [wp.transform.location.y for wp in route]
        plt.plot(x, y, marker="o", color="red")
        plt.title("Planned Route")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid()
        plt.savefig("planned_route.png", dpi=300)
        plt.close()

    @staticmethod
    def plot_road_network(graph):
        x = [graph[wp_id]["waypoint"].transform.location.x for wp_id in graph]
        y = [graph[wp_id]["waypoint"].transform.location.y for wp_id in graph]
        plt.scatter(x, y, s=1)
        plt.title("Road Network Graph")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis("equal")
        plt.grid()
        plt.savefig("road_network_graph.png", dpi=300)
        plt.close()
