import brain_state_calculate_c as bsc
import cpp_file_tools_c as cft
import ast
import numpy as np
import datetime
from Tkinter import *
import ttk
import tkFileDialog
from os.path import basename, splitext
from matplotlib import pyplot as plt
from sklearn import decomposition


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
        self.init_dir = ""
        self.use_HMM = IntVar()
        self.use_HMM.set(1)
        self.new_day = IntVar()
        self.new_day.set(0)
        self.stim_on = IntVar()
        self.stim_on.set(0)
        self.mod_chan_on = IntVar()
        self.mod_chan_on.set(0)
        self.include_classifier_res = IntVar()
        self.include_classifier_res.set(0)
        self.save_fig = IntVar()
        self.save_fig.set(0)
        self.show_fig = IntVar()
        self.show_fig.set(1)
        self.ext_img = StringVar()
        self.ext_img.set('.png')
        self.save_folder=""
        self.quantile_shrink = IntVar()
        self.quantile_shrink.set(0)

    def redirect_IO(self):
        sys.stdout = IORedirector(self.tb_stdout)
        sys.stderr = IORedirector(self.tb_stdout)

    def quit_app(self):
        self.destroy()
        self.quit()

    def load_classifier(self):
        self.my_bsc = bsc.brain_state_calculate(32, settings_path="classifierSettings.yaml")
        res = self.my_bsc.load_networks_file(self.init_dir)
        if res >= 0:
            self.enable_all_button()

        self.init_cft()
        self.update_param()

    def create_classifier(self):
        self.disable_all_button()
        self.my_bsc = bsc.brain_state_calculate(int(self.sb_weight_count_param.get())/int(self.sb_group_by_param.get()), settings_path="classifierSettings2.yaml")
        self.init_cft()
        self.update_classifier()
        res = self.my_bsc.init_networks_on_files(self.init_dir, self.my_cft, train_mod_chan=self.mod_chan_on.get(), autosave=True)
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
        if self.save_folder == "" or self.save_folder == "/":
            self.select_save_folder()
        self.update_classifier()
        self.my_bsc.test_classifier_on_file(self.my_cft, self.init_dir, on_modulate_chan=self.mod_chan_on.get(),
                                            gui=True, include_classifier_result=self.include_classifier_res.get(),
                                            save_folder=self.save_folder)
        self.enable_all_button()

    def plot_brain(self):
        self.disable_all_button()

        file_path = tkFileDialog.askopenfilename(multiple=True, initialdir=self.init_dir,  title="select cpp file to test the classifier", filetypes=[('all files', '.*'), ('text files', '.txt')])
        if file_path == "":
            self.enable_all_button()
            return -1

        paths = self.splitlist(file_path)
        for path in paths:
            l_res, l_obs = self.my_cft.read_cpp_files([path], use_classifier_result=False, cut_after_cue=False,
                                                      init_in_walk=True, on_stim=self.stim_on.get())
            if self.quantile_shrink.get():
                tmp_l_obs = []
                for obs in l_obs:
                    tmp_l_obs.append(self.my_bsc.obs_to_quantiles(obs, self.mod_chan_on.get()))
                l_obs = tmp_l_obs
            if len(l_obs) > 0:
                self.my_cft.plot_obs(l_obs, l_res, dir_path=self.save_folder,  extra_txt=splitext(basename(path))[0], gui=True)
            else:
                print "empty file"

        self.enable_all_button()

    def set_classifier_pca(self):
        nw = []
        for row in self.my_bsc.koho[0].network:
            for n in row:
                nw.append(n.weights)


        for row in self.my_bsc.koho[1].network:
            for n in row:
                nw.append(n.weights)

        nw = np.array(nw)

        self.pca = decomposition.PCA(n_components=2)
        self.pca.fit(nw)



    def plot_classifier_pca(self):
        try:
            self.pca == None
        except:
            self.set_classifier_pca()

        nw0 = []
        nw1 = []
        for row in self.my_bsc.koho[0].network:
            for n in row:
                nw0.append(n.weights)


        for row in self.my_bsc.koho[1].network:
            for n in row:
                nw1.append(n.weights)

        pca_nw0 = self.pca.transform(nw0)
        pca_nw1 = self.pca.transform(nw1)

        plt.figure()
        plt.scatter(pca_nw0[:, 0], pca_nw0[:, 1], marker='x', c='b', label="rest")
        plt.scatter(pca_nw1[:, 0], pca_nw1[:, 1], marker='x', c='g', label="walk")
        plt.legend()
        plt.show()

    def train_classifier(self):
        self.disable_all_button()
        self.update_classifier()
        training_method = self.sb_training.get()
        if training_method == "RL":
            self.my_bsc.train_on_files(self.init_dir, self.my_cft, is_healthy=False, new_day=self.new_day.get(), obs_to_add=int(self.sb_obs_to_add.get()), with_RL=True, train_mod_chan=self.mod_chan_on.get(), on_stim=self.stim_on.get(), autosave=True)
        elif training_method == "noRL":
            self.my_bsc.train_on_files(self.init_dir, self.my_cft, is_healthy=False, new_day=self.new_day.get(), obs_to_add=int(self.sb_obs_to_add.get()), with_RL=False, train_mod_chan=self.mod_chan_on.get(), on_stim=self.stim_on.get(), autosave=True)
        elif training_method == "unsupervised":
            self.my_bsc.train_unsupervised_on_files(self.init_dir, self.my_cft, is_healthy=False, obs_to_add=int(self.sb_obs_to_add.get()), train_mod_chan=self.mod_chan_on.get(), on_stim=self.stim_on.get(), autosave=True)
        #after training new_day go back to 0
        self.new_day.set(0)
        self.enable_all_button()

    def train_test(self):
        self.disable_all_button()
        if self.save_folder == "" or self.save_folder == "/":
            self.select_save_folder()
        self.update_classifier()
        file_path = tkFileDialog.askopenfilename(multiple=True, initialdir=self.init_dir,  title="select cpp file to test the classifier", filetypes=[('all files', '.*'), ('text files', '.txt')])
        if file_path == "":
            self.enable_all_button()
            print "no file selected"
            return -1

        paths = self.splitlist(file_path)
        for path in paths:
            l_res, l_obs = self.my_cft.read_cpp_files([path], use_classifier_result=False, cut_after_cue=False,
                                                      init_in_walk=True, on_stim=self.stim_on.get())
            if len(l_obs) > 0:
                #test
                success, l_of_res = self.my_bsc.test(l_obs, l_res, on_modulate_chan=self.mod_chan_on.get())
                if self.include_classifier_res.get() > 0:
                    l_res, l_obs = self.my_cft.read_cpp_files([path], use_classifier_result=True, cut_after_cue=False,
                                                              init_in_walk=True, on_stim=self.stim_on.get())
                    l_of_res["file_result"] = np.array(l_res).argmax(1)
                self.my_cft.plot_result(l_of_res, big_figure=False, dir_path=self.save_folder, extra_txt=splitext(basename(path))[0], gui=True)

                #train
                training_method = self.sb_training.get()
                if training_method == "RL":
                    self.my_bsc.train_one_file(path, self.my_cft, is_healthy=False, new_day=self.new_day.get(), obs_to_add=int(self.sb_obs_to_add.get()), with_RL=True, train_mod_chan=self.mod_chan_on.get(), on_stim=self.stim_on.get(), autosave=True)
                elif training_method == "noRL":
                    self.my_bsc.train_one_file(path, self.my_cft, is_healthy=False, new_day=self.new_day.get(), obs_to_add=int(self.sb_obs_to_add.get()), with_RL=False, train_mod_chan=self.mod_chan_on.get(), on_stim=self.stim_on.get(), autosave=True)
                elif training_method == "unsupervised":
                    self.my_bsc.train_unsupervised_one_file(path, self.my_cft, is_healthy=False, obs_to_add=int(self.sb_obs_to_add.get()), train_mod_chan=self.mod_chan_on.get(), on_stim=self.stim_on.get(), autosave=True)
                #after training new_day go back to 0
                self.new_day.set(0)
            else:
                print "empty file"

        self.enable_all_button()


    def update_param(self):
        self.e_HMM_param.delete(0, END)
        self.e_HMM_param.insert(0, str(self.my_bsc.A))

        self.sb_group_by_param.delete(0,"end")
        self.sb_group_by_param.insert(0, self.my_cft.group_chan)

        self.sb_weight_count_param.delete(0, "end")
        self.sb_weight_count_param.insert(0, self.my_cft.chan_count)

        self.sb_history_length_param.delete(0, "end")
        self.sb_history_length_param.insert(0, self.my_bsc.history_length)

        self.e_mod_chan_param.delete(0, END)
        self.e_mod_chan_param.insert(0, str(self.my_bsc.mod_chan))

        self.quantile_shrink.set(self.my_bsc.use_obs_quantile)
        self.t_quantile['text'] = str(self.my_bsc.qVec)

    def correct_list_string(self, text):
        return text.replace(' ', ',').replace('[,', '[').replace(',]', ']').replace(',,', ',').replace(',,', ',')

    def update_classifier(self):
        #convertion to obtain a correctly formated list from string list
        self.my_bsc.A = np.array(ast.literal_eval(self.correct_list_string(self.e_HMM_param.get())))
        self.my_bsc.weight_count = int(self.sb_weight_count_param.get()) / int(self.sb_group_by_param.get())
        self.my_bsc.history_length = int(self.sb_history_length_param.get())
        self.my_bsc.mod_chan = ast.literal_eval(self.correct_list_string(self.e_mod_chan_param.get()))
        self.my_bsc.use_quantile_shrink(self.quantile_shrink.get(), step=float(self.sb_quantile_step.get()))

        self.my_cft.group_chan = int(self.sb_group_by_param.get())
        self.my_cft.chan_count = int(self.sb_weight_count_param.get())
        self.my_cft.save = self.save_fig.get()
        self.my_cft.show = self.show_fig.get()
        self.my_cft.ext_img = self.ext_img.get()

    def change_dir(self):
        self.init_dir = tkFileDialog.askdirectory(initialdir=self.init_dir, mustexist=True)

    def select_save_folder(self):
        self.save_folder = tkFileDialog.askdirectory(initialdir=self.init_dir, mustexist=True, title="select the folder weher you want to save picture")
        self.save_folder += '/'
        self.t_save_folder['text'] = self.save_folder

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
        self.f_parameter.grid(row=0, column=0)

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
        self.e_HMM_param.delete(0, END)
        self.e_HMM_param.insert(0,str(np.array([[0.9, 0.1],[0.1,0.9]])))

        #mod chan
        self.t_mod_chan = Label(self.f_parameter, text="modulated channel")
        self.t_mod_chan.grid(row=5, column=0, sticky=E)
        self.e_mod_chan_param = Entry(self.f_parameter, width=30)
        self.e_mod_chan_param.grid(row=5, column=1, sticky=W)
        self.e_mod_chan_param.delete(0, END)
        self.e_mod_chan_param.insert(0, str(range(int(self.sb_weight_count_param.get())/int(self.sb_group_by_param.get()))))

        #quantile
        self.t_quantile_shrink = Label(self.f_parameter, text="quantile shrink")
        self.t_quantile_shrink.grid(row=6, column=0, sticky=E)
        self.cb_quantile_shrink = Checkbutton(self.f_parameter, variable=self.quantile_shrink)
        self.cb_quantile_shrink.grid(row=6, column=1, sticky=W)

        #quantile_step
        self.t_quantile_step = Label(self.f_parameter, text="quantile step")
        self.t_quantile_step.grid(row=7, column=0, sticky=E)
        self.sb_quantile_step = Spinbox(self.f_parameter, from_=0.05, to=1, width=5, increment=0.05, format="%.2f")
        self.sb_quantile_step.grid(row=7, column=1, sticky=W)

        #quantil vector
        self.t_quantile = Label(self.f_parameter, state=DISABLED)
        self.t_quantile.grid(row=8, columnspan=2)

        #update classifier parameter
        self.b_update_classifier = Button(self.f_parameter, text="update classifier", command=self.update_classifier, state=DISABLED)
        self.b_update_classifier.grid(row=9, columnspan=2)

        #train test
        self.f_train_test = LabelFrame(self.f_mainframe, padx=10, pady=10, text="Test and train parameter")
        self.f_train_test.grid(row=1, column=0)

        #train test parameter
        self.nb_train_test_param = ttk.Notebook(self.f_train_test)
        self.nb_train_test_param.grid(row=0, column=0, padx=10)

        #train parameter
        self.f_train_parameter = Frame(self.nb_train_test_param, padx=10, pady=10)
        self.f_train_parameter.grid(row=0, column=0)
        self.nb_train_test_param.add(self.f_train_parameter, text="Train")

        self.sb_training = Spinbox(self.f_train_parameter, values=("RL", "noRL", "unsupervised"), width=5)
        self.sb_training.grid(row=0, column=0, sticky=E)
        self.t_training_method = Label(self.f_train_parameter, text="training method")
        self.t_training_method.grid(row=0, column=1, sticky=W)

        self.sb_obs_to_add = Spinbox(self.f_train_parameter, from_=-10, to=10, width=5)
        self.sb_obs_to_add.grid(row=1, column=0, sticky=E)
        self.sb_obs_to_add.delete(0, END)
        self.sb_obs_to_add.insert(0, 0)
        self.t_obs_to_add = Label(self.f_train_parameter, text="obs to add")
        self.t_obs_to_add.grid(row=1, column=1, sticky=W)

        self.cb_use_HMM = Checkbutton(self.f_train_parameter, variable=self.use_HMM)
        self.cb_use_HMM.grid(row=2, column=0, sticky=E)
        self.t_use_HMM = Label(self.f_train_parameter, text="use HMM")
        self.t_use_HMM.grid(row=2, column=1, sticky=W)

        self.cb_new_day = Checkbutton(self.f_train_parameter, variable=self.new_day)
        self.cb_new_day.grid(row=3, column=0, sticky=E)
        self.t_new_day = Label(self.f_train_parameter, text="new day")
        self.t_new_day.grid(row=3, column=1, sticky=W)

        self.cb_stim_on = Checkbutton(self.f_train_parameter, variable=self.stim_on)
        self.cb_stim_on.grid(row=4, column=0, sticky=E)
        self.t_stim_on = Label(self.f_train_parameter, text="stim on only")
        self.t_stim_on.grid(row=4, column=1, sticky=W)

        self.cb_mod_chan_on = Checkbutton(self.f_train_parameter, variable=self.mod_chan_on)
        self.cb_mod_chan_on.grid(row=5, column=0, sticky=E)
        self.t_mod_chan_on = Label(self.f_train_parameter, text="train on modulated channel")
        self.t_mod_chan_on.grid(row=5, column=1, sticky=W)

        #test parameter
        self.f_test_parameter = Frame(self.nb_train_test_param, padx=10, pady=10)
        self.f_test_parameter.grid(row=0, column=0)
        self.nb_train_test_param.add(self.f_test_parameter, text="Test")

        self.cb_include_decoded = Checkbutton(self.f_test_parameter, variable=self.include_classifier_res)
        self.cb_include_decoded.grid(row=0, column=0, sticky=E)
        self.t_include_decoded = Label(self.f_test_parameter, text="include file results")
        self.t_include_decoded.grid(row=0, column=1, sticky=W)

        self.cb_show_fig = Checkbutton(self.f_test_parameter, variable=self.show_fig, command=self.on_check_save_fig)
        self.cb_show_fig.grid(row=1, column=0, sticky=E)
        self.t_show_fig = Label(self.f_test_parameter, text="show figure")
        self.t_show_fig.grid(row=1, column=1, sticky=W)

        self.cb_save_fig = Checkbutton(self.f_test_parameter, variable=self.save_fig, command=self.on_check_save_fig)
        self.cb_save_fig.grid(row=2, column=0, sticky=E)
        self.t_save_fig = Label(self.f_test_parameter, text="save figure")
        self.t_save_fig.grid(row=2, column=1, sticky=W)

        self.b_save_folder = Button(self.f_test_parameter, text="Saving folder", command=self.select_save_folder, state=DISABLED)
        self.b_save_folder.grid(row=3, column=0, sticky=E)
        self.t_save_folder = Label(self.f_test_parameter, height=1, width=20, state=DISABLED, anchor=W)
        self.t_save_folder.grid(row=3, column=1, sticky=W)

        self.rb_ext_img_eps = Radiobutton(self.f_test_parameter, variable=self.ext_img, value=".eps", state=DISABLED)
        self.rb_ext_img_eps.grid(row=4, column=0, sticky=E)
        self.t_ext_img_eps = Label(self.f_test_parameter, text='.eps', state=DISABLED)
        self.t_ext_img_eps.grid(row=4, column=1, sticky=W)

        self.rb_ext_img_png = Radiobutton(self.f_test_parameter, variable=self.ext_img, value=".png", state=DISABLED)
        self.rb_ext_img_png.grid(row=5, column=0, sticky=E)
        self.t_ext_img_png = Label(self.f_test_parameter, text='.png', state=DISABLED)
        self.t_ext_img_png.grid(row=5, column=1, sticky=W)

        #train test button
        self.f_train_test_button = Frame(self.f_train_test)
        self.f_train_test_button.grid(row=0, column=1)

        self.b_test = Button(self.f_train_test_button, text="test classifier", command=self.test_classifier, state=DISABLED)
        self.b_test.grid(row=0, column=0, pady=10)

        #plot obs
        self.b_obs = Button(self.f_train_test_button, text="plot brain", command=self.plot_brain, state=DISABLED)
        self.b_obs.grid(row=1, column=0, pady=10)

        #set classifier pca
        self.b_set_pca_classifier = Button(self.f_train_test_button, text= "set classifier pca", command=self.set_classifier_pca, state=DISABLED)
        self.b_set_pca_classifier.grid(row=2, column=0, pady=10)

        #plot classifier pca
        self.b_pca_classifier = Button(self.f_train_test_button, text="plot classifier pca", command=self.plot_classifier_pca, state=DISABLED)
        self.b_pca_classifier.grid(row=3, column=0, pady=10)

        #train button
        self.b_train = Button(self.f_train_test_button, text="train classifier", command=self.train_classifier, state=DISABLED)
        self.b_train.grid(row=4, column=0, pady=10)

        self.b_train_test = Button(self.f_train_test_button, text="train and test", command=self.train_test, state=DISABLED)
        self.b_train_test.grid(row=5, column=0, pady=10)

    def enable_all_button(self):
        self.filemenu.entryconfig(2, state=NORMAL)
        self.b_test['state'] = NORMAL
        self.b_train['state'] = NORMAL
        self.b_obs['state'] = NORMAL
        self.b_train_test['state'] = NORMAL
        self.b_update_classifier['state'] = NORMAL
        self.b_pca_classifier['state'] = NORMAL
        self.b_set_pca_classifier['state'] = NORMAL
        print "# # DONE # #"

    def disable_all_button(self):
        self.b_test['state'] = DISABLED
        self.b_train['state'] = DISABLED
        self.b_obs['state'] = DISABLED
        self.b_train_test['state'] = DISABLED
        self.b_update_classifier['state'] = DISABLED
        self.b_pca_classifier['state'] = DISABLED
        self.b_set_pca_classifier['state'] = DISABLED

    def on_check_save_fig(self):
        if self.save_fig.get() == ON:
            self.b_save_folder['state'] = NORMAL
            self.rb_ext_img_eps['state'] = NORMAL
            self.t_ext_img_eps['state'] = NORMAL
            self.rb_ext_img_png['state'] = NORMAL
            self.t_ext_img_png['state'] = NORMAL
        elif self.save_fig.get() == OFF:
            self.b_save_folder['state'] = DISABLED
            self.rb_ext_img_eps['state'] = DISABLED
            self.t_ext_img_eps['state'] = DISABLED
            self.rb_ext_img_png['state'] = DISABLED
            self.t_ext_img_png['state'] = DISABLED

app = Classifier_GUI(None)
app.mainloop()