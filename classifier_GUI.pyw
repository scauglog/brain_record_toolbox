import brain_state_calculate as bsc
import cpp_file_tools as cft
import ast
import numpy as np
import datetime
import sys
from Tkinter import *
import tkFileDialog

class IORedirector(object):
    '''A general class for redirecting I/O to this Text widget.'''
    def __init__(self, text_area):
        self.text_area = text_area

    def write(self, str):
        self.text_area.insert(END, str)
        # force widget to display the end of the text (follow the input)
        self.text_area.see(END)
        # force refresh of the widget to be sure that thing are displayed
        self.text_area.update_idletasks()

class Classifier_GUI(Tk):
    def __init__(self, parent):
        Tk.__init__(self, parent)
        self.parent = parent
        self.title("Classifier config")
        self.protocol("WM_DELETE_WINDOW", self.quit_app)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.grid()
        self.init_variable()
        self.init_window()
        self.redirect_IO()

    def init_variable(self):
        self.my_bsc = None
        self.my_cft = None
        self.init_dir = "C:\\"
        self.use_HMM = IntVar()
        self.use_HMM.set(1)
        self.new_day = IntVar()
        self.new_day.set(0)
        self.stim_on = IntVar()
        self.mod_chan_on = IntVar()

    def redirect_IO(self):
        sys.stdout = IORedirector(self.tb_stdout)
        sys.stderr = IORedirector(self.tb_stdout)

    def quit_app(self):
        self.destroy()
        self.quit()

    def load_classifier(self):
        self.my_bsc = bsc.brain_state_calculate(32)
        res = self.my_bsc.load_networks_file(self.init_dir)
        if res >= 0:
            self.enable_all_button()
        self.update_param()
        self.init_cft()

    def create_classifier(self):
        self.disable_all_button()
        self.my_bsc = bsc.brain_state_calculate(32)
        self.init_cft()
        res = self.my_bsc.init_networks_on_files(self.init_dir, self.my_cft)
        print res
        if res >= 0:
            self.enable_all_button()
        self.update_param()

    def init_cft(self):
        self.my_cft = cft.cpp_file_tools(int(self.sb_weight_count_param.get()), int(self.sb_group_by_param.get()))
        self.my_cft.show = True

    def save_classifier(self):
        self.disable_all_button()
        self.update_classifier()
        self.my_bsc.save_networks_on_file(self.init_dir, datetime.date.today().strftime("%Y%m%d"))
        self.enable_all_button()

    def test_classifier(self):
        self.disable_all_button()
        self.update_classifier()
        file_path = tkFileDialog.askopenfilename(multiple=True, initialdir=self.init_dir,  title="select cpp file to test the classifier", filetypes=[('all files', '.*'), ('text files', '.txt')])
        if file_path == "":
            self.enable_all_button()
            return -1

        paths = self.splitlist(file_path)
        for path in paths:
            l_res, l_obs = self.my_cft.read_cpp_files([path], is_healthy=False, cut_after_cue=False, init_in_walk=True, on_stim=self.stim_on.get())
            if len(l_obs) > 0:
                success, l_of_res = self.my_bsc.test(l_obs, l_res, on_modulate_chan=self.mod_chan_on.get())
                self.my_cft.plot_result(l_of_res, big_figure=False)
            else:
                print "empty file"

        self.enable_all_button()
        self.my_cft.show_fig()

    def plot_brain(self):
        self.disable_all_button()

        file_path = tkFileDialog.askopenfilename(multiple=True, initialdir=self.init_dir,  title="select cpp file to test the classifier", filetypes=[('all files', '.*'), ('text files', '.txt')])
        if file_path == "":
            self.enable_all_button()
            return -1

        paths = self.splitlist(file_path)
        for path in paths:
            l_res, l_obs = self.my_cft.read_cpp_files([path], is_healthy=False, cut_after_cue=False, init_in_walk=True, on_stim=self.stim_on.get())
            if len(l_obs) > 0:
                self.my_cft.plot_obs(l_obs, l_res)
            else:
                print "empty file"

        self.enable_all_button()
        self.my_cft.show_fig()

    def train_classifier(self):
        self.disable_all_button()
        self.update_classifier()
        training_method = self.sb_training.get()
        if training_method == "RL":
            self.my_bsc.train_on_files(self.init_dir, self.my_cft, is_healthy=False, new_day=self.new_day.get(), obs_to_add=int(self.sb_obs_to_add.get()), with_RL=True, train_mod_chan=self.mod_chan_on.get(), on_stim=self.stim_on.get())
        elif training_method == "noRL":
            self.my_bsc.train_on_files(self.init_dir, self.my_cft, is_healthy=False, new_day=self.new_day.get(), obs_to_add=int(self.sb_obs_to_add.get()), with_RL=False, train_mod_chan=self.mod_chan_on.get(), on_stim=self.stim_on.get())
        elif training_method == "unsupervised":
            self.my_bsc.train_unsupervised_on_files(self.init_dir, self.my_cft, is_healthy=False, obs_to_add=int(self.sb_obs_to_add.get()), train_mod_chan=self.mod_chan_on.get(), on_stim=self.stim_on.get())
        #after training new_day go back to 0
        self.new_day.set(0)
        self.enable_all_button()

    def update_param(self):
        self.e_HMM_param.delete(0, END)
        self.e_HMM_param.insert(0, str(self.my_bsc.A))

        self.sb_weight_count_param.delete(0, "end")
        self.sb_weight_count_param.insert(0, self.my_bsc.weight_count)

        self.sb_history_length_param.delete(0, "end")
        self.sb_history_length_param.insert(0, self.my_bsc.history_length)

        self.e_mod_chan_param.delete(0, END)
        self.e_mod_chan_param.insert(0, str(self.my_bsc.mod_chan))

    def correct_list_string(self, text):
        return text.replace(' ', ',').replace('[,', '[').replace(',]', ']').replace(',,', ',').replace(',,', ',')

    def update_classifier(self):
        #convertion to obtain a correctly formated list from string list
        self.my_bsc.A = np.array(ast.literal_eval(self.correct_list_string(self.e_HMM_param.get())))
        self.my_bsc.weight_count = int(self.sb_weight_count_param.get())
        self.my_bsc.history_length = int(self.sb_history_length_param.get())
        self.my_bsc.mod_chan = ast.literal_eval(self.correct_list_string(self.e_mod_chan_param.get()))

    def change_dir(self):
        self.init_dir = tkFileDialog.askdirectory(initialdir=self.init_dir, mustexist=True)

    def init_window(self):
        #menubar
        self.menubar = Menu(self)
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="New", command=self.create_classifier)
        self.filemenu.add_command(label="Open", command=self.load_classifier)
        self.filemenu.add_command(label="Save as...", command=self.save_classifier, state=DISABLED)
        self.filemenu.add_command(label="change dir", command=self.change_dir)
        self.menubar.add_cascade(label="File", menu=self.filemenu)
        self.config(menu=self.menubar)

        self.f_mainframe = Frame(self, padx=10, pady=10)
        self.f_mainframe.pack(fill=X)
        #parameter of the classifier
        self.f_parameter = LabelFrame(self.f_mainframe, text="Classifier parameter", padx=10,pady=10)
        self.f_parameter.grid(row=0,column=0)

        self.tb_stdout = Text(self.f_mainframe)
        self.tb_stdout.grid(row=0, column=1, rowspan=2)

        #group by
        self.t_group_by = Label(self.f_parameter, text="group by")
        self.t_group_by.grid(row=1, column=0, sticky=E)
        self.sb_group_by_param = Spinbox(self.f_parameter, from_=1, to=256, width=5)
        self.sb_group_by_param.grid(row=1, column=1, sticky=W)

        #weight count
        self.t_weight_count = Label(self.f_parameter, text="weight count")
        self.t_weight_count.grid(row=2, column=0, sticky=E)
        self.sb_weight_count_param = Spinbox(self.f_parameter, from_=0, to=256, width=5)
        self.sb_weight_count_param.grid(row=2, column=1, sticky=W)
        self.sb_weight_count_param.delete(0, "end")
        self.sb_weight_count_param.insert(0, 32)

        #history length
        self.t_history_length = Label(self.f_parameter, text="history length")
        self.t_history_length.grid(row=3, column=0, sticky=E)
        self.sb_history_length_param = Spinbox(self.f_parameter, from_=1, to=10, width=5)
        self.sb_history_length_param.grid(row=3, column=1, sticky=W)

        #HMM
        self.t_HMM = Label(self.f_parameter, text="HMM parameter")
        self.t_HMM.grid(row=4, column=0, sticky=E)
        self.e_HMM_param = Entry(self.f_parameter, width=30)
        self.e_HMM_param.grid(row=4, column=1, sticky=W)

        #mod chan
        self.t_mod_chan = Label(self.f_parameter, text="modulated channel")
        self.t_mod_chan.grid(row=5, column=0, sticky=E)
        self.e_mod_chan_param = Entry(self.f_parameter, width=30)
        self.e_mod_chan_param.grid(row=5, column=1, sticky=W)

        #train test
        self.f_train_test = LabelFrame(self.f_mainframe, padx=10, pady=10, text="Test and train parameter")
        self.f_train_test.grid(row=1,column=0)

        #train test parameter
        self.f_train_test_parameter = Frame(self.f_train_test)
        self.f_train_test_parameter.grid(row=1, column=0)

        self.sb_training = Spinbox(self.f_train_test_parameter, values=("RL", "noRL", "unsupervised"), width=5)
        self.sb_training.grid(row=0, column=0, sticky=E)
        self.t_training_method = Label(self.f_train_test_parameter, text="training method")
        self.t_training_method.grid(row=0, column=1, sticky=W)

        self.sb_obs_to_add = Spinbox(self.f_train_test_parameter, from_=-10, to=10, width=5)
        self.sb_obs_to_add.grid(row=1, column=0, sticky=E)
        self.sb_obs_to_add.delete(0, END)
        self.sb_obs_to_add.insert(0, 0)
        self.t_obs_to_add = Label(self.f_train_test_parameter, text="obs to add")
        self.t_obs_to_add.grid(row=1, column=1, sticky=W)

        self.cb_use_HMM = Checkbutton(self.f_train_test_parameter, variable=self.use_HMM)
        self.cb_use_HMM.grid(row=2, column=0, sticky=E)
        self.t_use_HMM = Label(self.f_train_test_parameter, text="use HMM")
        self.t_use_HMM.grid(row=2, column=1, sticky=W)

        self.cb_new_day = Checkbutton(self.f_train_test_parameter, variable=self.new_day)
        self.cb_new_day.grid(row=3, column=0, sticky=E)
        self.t_new_day = Label(self.f_train_test_parameter, text="new day")
        self.t_new_day.grid(row=3, column=1, sticky=W)

        self.cb_stim_on = Checkbutton(self.f_train_test_parameter, variable=self.stim_on)
        self.cb_stim_on.grid(row=4, column=0, sticky=E)
        self.t_stim_on = Label(self.f_train_test_parameter, text="stim on only")
        self.t_stim_on.grid(row=4, column=1, sticky=W)

        self.cb_mod_chan_on = Checkbutton(self.f_train_test_parameter, variable=self.mod_chan_on)
        self.cb_mod_chan_on.grid(row=5, column=0, sticky=E)
        self.t_mod_chan_on = Label(self.f_train_test_parameter, text="train on modulated channel")
        self.t_mod_chan_on.grid(row=5, column=1, sticky=W)

        #train test button
        self.f_train_test_button = Frame(self.f_train_test)
        self.f_train_test_button.grid(row=1, column=1)

        self.b_test = Button(self.f_train_test_button, text="test classifier", command=self.test_classifier, state=DISABLED)
        self.b_test.grid(row=0, column=0, pady=10)

        #plot obs
        self.b_obs = Button(self.f_train_test_button, text="plot brain", command=self.plot_brain, state=DISABLED)
        self.b_obs.grid(row=1, column=0, pady=10)

        #train button
        self.b_train = Button(self.f_train_test_button, text="train classifier", command=self.train_classifier, state=DISABLED)
        self.b_train.grid(row=2, column=0, pady=10)



    def enable_all_button(self):
        self.filemenu.entryconfig(2, state=NORMAL)
        self.b_test['state'] = NORMAL
        self.b_train['state'] = NORMAL
        self.b_obs['state'] = NORMAL

    def disable_all_button(self):
        self.b_test['state'] = DISABLED
        self.b_train['state'] = DISABLED
        self.b_obs['state'] = DISABLED

app = Classifier_GUI(None)
app.mainloop()