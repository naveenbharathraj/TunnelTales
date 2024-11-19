import os
import tkinter as tk
from tkinter import ttk, filedialog as fd, messagebox
from ttkthemes import ThemedStyle, ThemedTk

from src.solver_configuration.solver_configuration import SolverConfiguration
from src import gui_support
import main

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import scipy.io as spio

import logging
import threading

def open_file_dialog(is_config):
    if is_config:
        title="Select the Configuration File"
        filetypes=(("cfg", "*.cfg"), ("all files", "*.*"))
    else:
        title="Select the Output MAT File"
        filetypes=(("mat", "*.mat"), ("all files", "*.*"))
        
    filename = fd.askopenfilename(
        initialdir=os.getcwd(),
        title=title,
        filetypes=filetypes
    )
    return filename

def create_new_file():
    def save_to_file():
        gui_support.get_user_input(user_entries)

    def load_from_file():
        loaded_entries = gui_support.load_file_and_display(new_window)
        if loaded_entries:
            user_entries.update(loaded_entries)

    new_window = tk.Toplevel(root)
    new_window.title("Window One")

    user_input_frame = ttk.Frame(new_window, padding="10")
    user_input_frame.pack(fill=tk.BOTH, expand=True)

    user_entries = gui_support.create_gui_with_values_and_dropdowns(user_input_frame)

    label_text = "Create New Configuration"
    label = ttk.Label(user_input_frame, text=label_text, font=('Helvetica', 16))
    label.grid(row=0, column=0, columnspan=2, pady=10)

    button_save = ttk.Button(user_input_frame, text="Save to File", command=save_to_file, style='TButton')
    button_save.grid(row=2, column=0, columnspan=2, pady=20)
    button_load = ttk.Button(user_input_frame, text="Load a File", command=load_from_file, style='TButton')
    button_load.grid(row=2, column=1, columnspan=2, pady=20)

    new_window.grid_columnconfigure(0, weight=1)
    new_window.grid_rowconfigure(2, weight=1)

def load_file_and_run():
    def run_simulation():
        nonlocal filename
        nonlocal progress_bar

        try:
            main.main_run(filename, logging)
            tk.messagebox.showinfo(title='Done', message='Simulation Completed.')
        except Exception as e:
            tk.messagebox.showerror(title='Error', message=f'Error during simulation: {str(e)}')
        finally:
            progress_bar.stop()

    def check_thread():
        if thread.is_alive():
            root.after(100, check_thread)
        else:
            progress_bar.stop()
            progress_bar.destroy()

    try:
        filename = open_file_dialog(True)

        if filename:
            progress_bar = ttk.Progressbar(root, mode='indeterminate')
            progress_bar.pack(padx=10, pady=10)
            progress_bar.start()

            # Start a new thread to run the simulation
            thread = threading.Thread(target=run_simulation)
            thread.start()

            # Periodically check if the thread has finished
            root.after(100, check_thread)
        else:
            return
    except FileNotFoundError as e:
        messagebox.showerror(title='Error', message=f'File not found: {str(e)}')
    except Exception as e:
        messagebox.showerror(title='Error', message=f'Error during file selection: {str(e)}')


def view_train_plot():
    def update_graph():
        try:
            t = t_entry.get()
            print("ee")
            gui_support.view_train_position(canvas, ax1, ax2, ax3, solver_config, t ,logging)
        except Exception as e:
            logging.error(f"An error occurred in update_graph train graph: {e}")

    try:
        filename = open_file_dialog(True)

        if not filename:
            return

        new_window = tk.Toplevel(root)
        new_window.title("Window Three")

        frame_entries = ttk.Frame(new_window)
        frame_entries.pack(fill=tk.BOTH, expand=True)

        logging.info(f"Reading configuration from {filename}")
        solver_config = SolverConfiguration(filename, logging, False)
        logging.info(f"Read sucessfully")
        
        label_t = ttk.Label(frame_entries, text="T Value:", font=("Arial", 16))
        label_t.grid(row=0, column=0, padx=10, pady=10, sticky='e')

        t_entry = ttk.Entry(frame_entries, font=("Arial", 16))
        t_entry.grid(row=0, column=1, padx=10, pady=10, sticky='w')

        add_button = ttk.Button(
            frame_entries, text="Refresh Graph", command=update_graph, style='TButton')
        add_button.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')
        
        frame_entries.columnconfigure(0, weight=1)  # Column 0 expands horizontally
        frame_entries.columnconfigure(1, weight=1)  # Column 1 expands horizontally
        frame_entries.rowconfigure(0, weight=1)     # Row 0 expands vertically
        frame_entries.rowconfigure(1, weight=1)     # Row 1 expands vertically

        frame_plot = ttk.Frame(new_window)
        frame_plot.pack(fill=tk.BOTH, expand=True)

        fig = Figure(figsize=(20, 15))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        fig.subplots_adjust(hspace=1)

        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        toolbar = NavigationToolbar2Tk(canvas, frame_plot)
        toolbar.update()

        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    except FileNotFoundError:
        logging.error("File not found. Please provide the correct file path.")
    except Exception as e:
        logging.error(f"An error occurred in plotting train graph: {e}")

def view_results():
    
    def update_graph():
        try:
            position = float(dropdown_var.get())
            position_index=options.index(position)
            apply_filter=apply_filter_var.get()
            
            gui_support.view_train_results(canvas,ax,position, simulation_data['t'][0] , simulation_data['p_history'][position_index], apply_filter)
        except Exception as e:
            logging.error(f"An error occurred in update_graph train graph: {e}")
            
    try:
        filename = open_file_dialog(False)
        #filename=1

        if not filename:
            return
        
        new_window = tk.Toplevel(root)
        new_window.title("Window Four")

        frame_entries = ttk.Frame(new_window)
        frame_entries.pack(fill=tk.BOTH, expand=False)
        
        logging.info(f"Reading configuration from {filename}")
        simulation_data=spio.loadmat(filename)
        
        simulation_data=spio.loadmat('output/p_history.mat')
        label_t = ttk.Label(frame_entries, text="Probe Location :", font=("Arial", 16))
        label_t.grid(row=2, column=0, pady=10, sticky='w')
        
        options = simulation_data['x_probe'][0].tolist()
        dropdown_var = tk.StringVar(frame_entries)
        #dropdown_var.set(options[0])

        dropdown = ttk.Combobox(frame_entries, textvariable=dropdown_var, values=options)
        dropdown.grid(row=2, column=1, pady=10,sticky='w')
        
        apply_filter_var = tk.BooleanVar()
        checkbox = ttk.Checkbutton(frame_entries, text="Apply Filter", variable=apply_filter_var)
        checkbox.grid(row=2, column=2, padx=10,sticky='w')

        add_button = ttk.Button(
            frame_entries, text="Refresh Graph", command=update_graph)
        add_button.grid(row=2, column=3, padx=10,sticky='w')
        
        frame_entries.columnconfigure(0, weight=1)  # Column 0 expands horizontally
        frame_entries.columnconfigure(1, weight=1)  # Column 1 expands horizontally
        frame_entries.rowconfigure(0, weight=1)     # Row 0 expands vertically
        frame_entries.rowconfigure(1, weight=1)     # Row 1 expands vertically
        
        frame_plot = ttk.Frame(new_window)
        frame_plot.pack(fill=tk.BOTH, expand=True)  # Expand the frame to fill both X and Y directions

        fig = Figure(figsize=(20, 15))
        ax = fig.add_subplot(111)  # Create a single subplot

        canvas = FigureCanvasTkAgg(fig, master=frame_plot)
        toolbar = NavigationToolbar2Tk(canvas, frame_plot)
        toolbar.update()

        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    except FileNotFoundError:
        logging.error("File not found. Please provide the correct file path.")
    except Exception as e:
        logging.error(f"An error occurred in plotting train graph: {e}")


def toggle_theme():
    current_theme = style.theme_use()
    new_theme = 'arc' if current_theme != 'arc' else 'equilux'
    change_theme(new_theme)

def change_theme(theme_name):
    style.theme_use(theme_name)

if __name__ == "__main__":
    logging.basicConfig(filename='log/gui.log', filemode='w',
                        format='%(asctime)s - %(message)s', level=logging.INFO)

    root = ThemedTk(theme="arc")
    root.title("Main Window")
    window_width = 600
    window_height = 400
    root.geometry(f"{window_width}x{window_height}")

    frame = ttk.Frame(root, padding="10")
    frame.pack(fill=tk.BOTH, expand=True)

    label = ttk.Label(frame, text="Welcome to Your GUI!\nSelect an action below:")
    label.pack(pady=12)

    button1 = ttk.Button(frame, text="Create New Configuration File", command=create_new_file, width=40, style='TButton')
    button1.pack(pady=12)

    button2 = ttk.Button(frame, text="Load Configuration File and Run", command=load_file_and_run, width=40, style='TButton')
    button2.pack(pady=20)

    button3 = ttk.Button(frame, text="View Train in the Track", command=view_train_plot, width=40, style='TButton')
    button3.pack(pady=12)

    button4 = ttk.Button(frame, text="View Results", command=view_results, width=40, style='TButton')
    button4.pack(pady=12)

    style = ThemedStyle(root)
    initial_theme = 'arc'
    style.theme_use(initial_theme)

    toggle_button = ttk.Button(root, text="Toggle Theme", command=toggle_theme, style='TButton')
    toggle_button.place(x=10, y=10)

    root.mainloop()
