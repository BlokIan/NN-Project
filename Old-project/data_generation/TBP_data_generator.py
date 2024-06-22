import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Button

def generate_data(n):
    for j in range(n):
        valid_conditions = False

        while not valid_conditions:
            # Time step [s]
            step = 0.001

            # Body Mass
            m1 = random.uniform(0.8, 1.2)
            m2 = random.uniform(0.8, 1.2)
            m3 = random.uniform(0.8, 1.2)
            # m1 = 1
            # m2 = 1
            # m3 = 1
            # print(m1, m2, m3)
            M = np.array([m1, m2, m3])

            # Newton constant
            G = 1

            # Time vector
            t = np.arange(0, 5 + step, step) # creates vector (list) of values between 0 and 5 with steps of "step"

            # Initial conditions -- positions
            # These initial parameters may require some aditional changes for interesting data generation (body mass is also subject for change)
            x10 = 0 # -1
            y10 = 1 # 0
            # print(x10, y10)
            x20 = random.uniform(-1, 0)
            y20 = random.uniform(-1, 0) 

            x30 = -x20
            y30 = -y20

            # Initial conditions -- velocities
            vx10 = 0.3471
            vy10 = 0.5327

            vx20 = vx10
            vy20 = vy10

            vx30 = -2 * vx10
            vy30 = -2 * vy10

            def compute_total_energy(x, y, vx, vy, M, G):
                kinetic_energy = 0.5 * M[0] * (vx[0]**2 + vy[0]**2) + \
                                 0.5 * M[1] * (vx[1]**2 + vy[1]**2) + \
                                 0.5 * M[2] * (vx[2]**2 + vy[2]**2)
                
                r12 = np.sqrt((x[0] - x[1])**2 + (y[0] - y[1])**2)
                r13 = np.sqrt((x[0] - x[2])**2 + (y[0] - y[2])**2)
                r23 = np.sqrt((x[1] - x[2])**2 + (y[1] - y[2])**2)
                
                potential_energy = -G * M[0] * M[1] / r12 - G * M[0] * M[2] / r13 - G * M[1] * M[2] / r23
                
                total_energy = kinetic_energy + potential_energy
                return total_energy

            initial_positions = np.array([[x10, y10], [x20, y20], [x30, y30]])
            initial_velocities = np.array([[vx10, vy10], [vx20, vy20], [vx30, vy30]])

            total_energy = compute_total_energy(initial_positions[:, 0], initial_positions[:, 1],
                                                initial_velocities[:, 0], initial_velocities[:, 1], M, G)

            if total_energy < 0:
                valid_conditions = True

            # Preparing the vector to store the solution
            x = np.zeros((len(t), 3))
            y = np.zeros((len(t), 3))
            vx = np.zeros((len(t), 3))
            vy = np.zeros((len(t), 3))

            ax = np.zeros((3, 3))
            ay = np.zeros((3, 3))

            # Assign the initial condition to the first element of the solution vectors
            x[0] = [x10, x20, x30]
            y[0] = [y10, y20, y30]
            vx[0] = [vx10, vx20, vx30]
            vy[0] = [vy10, vy20, vy30]

            def compute_a(x, y, M, ax, ay, G):
                
                # Compute the distance between current body and the others ( also with itself )
                for j in range(3):
                    dx = x[j] - x
                    dy = y[j] - y
                    ax[:, j] = (-dx * M * G) / (np.sqrt(dx**2 + dy**2)**3)
                    ay[:, j] = (-dy * M * G) / (np.sqrt(dx**2 + dy**2)**3)

                # change NaN into zeros
                ax[np.isnan(ax)] = 0
                ay[np.isnan(ay)] = 0
                
                ax_tot = np.sum(ax, axis=0)
                ay_tot = np.sum(ay, axis=0)
                
                return ax_tot, ay_tot

            # Simple Euler method
            for i in range(len(t) - 1):
                ax_tot, ay_tot = compute_a(x[i], y[i], M, ax, ay, G)
                x[i+1] = x[i] + step * vx[i]
                y[i+1] = y[i] + step * vy[i]
                vx[i+1] = vx[i] + step * ax_tot
                vy[i+1] = vy[i] + step * ay_tot

            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin to make space for the button
            ax.plot(x[:, 0], y[:, 0], 'r')
            ax.plot(x[:, 1], y[:, 1], 'g')
            ax.plot(x[:, 2], y[:, 2], 'b')
            ax.plot(x10, y10, 'ro', markerfacecolor='r')
            ax.plot(x20, y20, 'go', markerfacecolor='g')
            ax.plot(x30, y30, 'bo', markerfacecolor='b')
            ax.legend(['Body 1', 'Body 2', 'Body 3'])
            ax.set_title('TBP')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.grid(True)

            def on_save_button_click(event):
                data = np.column_stack((t, x, y, vx, vy))
                header = f'm1 {m1} m2 {m2} m3 {m3} \n t x1 x2 x3 y1 y2 y3 vx1 vx2 vx3 vy1 vy2 vy3'
                np.savetxt(f'training_data/data{j}.txt', data, header=header)
                print("Data saved")
                plt.close()

            # Add a button to save the data
            save_button_ax = plt.axes([0.7, 0.05, 0.2, 0.075])
            save_button = Button(save_button_ax, 'Save Data')
            save_button.on_clicked(on_save_button_click)

            plt.show()

generate_data(10)