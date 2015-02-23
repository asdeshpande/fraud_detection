#!/usr/bin/python
import Tkinter as tk
import tkMessageBox
import os

out = []
desc = ""

class Application(tk.Frame):
    r1 = 1
    PATH = os.environ['PWD'] 
    ALGPATH = "" 
    OUTFILE = ""

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack(expand=tk.YES, fill=tk.BOTH)
        self.createWidgets()


    def createWidgets(self):
	self.r1 = tk.IntVar()
	#self.r1.set(1)
	self.label0 = tk.Label(text="Choose algorithm:")
        self.label0.pack()

	self.R1 = tk.Radiobutton(root, text="MLFF", variable=self.r1, value=1,command = self.sel)
	self.R2 = tk.Radiobutton(root, text="K-means Clustering", variable=self.r1, value=2,command = self.sel)

	self.R1.pack(side="top")
	self.R2.pack(side="top")

        self.START = tk.Button(self)
        self.START["text"] = "Start"
        self.START["command"] = self.run_algo
        self.START.pack(side="top")
	    
        self.QUIT = tk.Button(self, text="QUIT", fg="red", command=root.destroy)
        self.QUIT.pack(side="bottom")

	termf = tk.Frame(self, height=56, width=20)
	termf.pack()
	wid = termf.winfo_id()
	
          
    def sel(self):
        if self.r1.get() == 1:
		self.ALGPATH = 'xterm -into %d -geometry 56x20 -sb -hold -e /usr/bin/python new_mlff.py &' 
		self.OUTFILE = "mlout.dat"
		desc = '''Multi-Layer Feed Forward Network
Learning-Rate : 	0.01
Input Neurons : 	9
Hidden Neurons:		14
Output Neurons:		2
'''
		
		T.insert(tk.END,desc)

	elif self.r1.get() == 2:
		self.ALGPATH = 'xterm -into %d -geometry 56x20 -sb -hold -e /usr/bin/python kmeans.py &' 
		self.OUTFILE = "pnout.dat"
	else:
		print "Choose algorithm:"

	
    def run_algo(self):
	cmd = self.ALGPATH + ">" + self.OUTFILE
	res = os.system(cmd)
	    
      
root = tk.Tk()
app = Application(master=root)
app.master.title("final_project")
app.master.minsize(350,250)
app.master.maxsize(350,250)
T = tk.Text(app, height = 5, width = 120)
T.pack()
app.mainloop()
