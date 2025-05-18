import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import Value

def run_simulation(risk_flag):
    pos = np.array([0.5, 0.5])
    fig, ax = plt.subplots()
    circle = plt.Circle(pos, 0.05, color='green')
    ax.add_patch(circle)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    plt.title("Robot Movement Simulation")

    while True:
        circle.remove()
        if risk_flag.value == 0:
            movement = (np.random.rand(2) - 0.5) * 0.1
            pos = np.clip(pos + movement, 0.05, 0.95)
            circle = plt.Circle(pos, 0.05, color='green')
        else:
            circle = plt.Circle(pos, 0.05, color='red')

        ax.add_patch(circle)
        plt.draw()
        plt.pause(0.5)

if __name__ == "__main__":
    from multiprocessing import Process, Value
    risk_flag = Value('i', 0)

    from test import camera_loop  # Assumes cam.py is in same dir

    cam_proc = Process(target=camera_loop, kwargs={'risk_flag': risk_flag})
    sim_proc = Process(target=run_simulation, args=(risk_flag,))

    cam_proc.start()
    sim_proc.start()

    cam_proc.join()
    sim_proc.join()