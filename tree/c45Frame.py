# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version Jun  5 2014)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc
import os
from C45 import *

###########################################################################
## Class MyFrame1
###########################################################################

class CFrame(wx.Frame):
	def __init__(self, parent):
		wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=wx.EmptyString, pos=wx.DefaultPosition,
						  size=wx.Size(500, 300), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

		self.SetSizeHintsSz(wx.DefaultSize, wx.DefaultSize)

		bSizer1 = wx.BoxSizer(wx.VERTICAL)

		fgSizer1 = wx.FlexGridSizer(1, 4, 0, 0)
		fgSizer1.SetFlexibleDirection(wx.BOTH)
		fgSizer1.SetNonFlexibleGrowMode(wx.FLEX_GROWMODE_SPECIFIED)

		self.m_staticText2 = wx.StaticText(self, wx.ID_ANY, u"filepath", wx.DefaultPosition, wx.DefaultSize, 0)
		self.m_staticText2.Wrap(-1)
		fgSizer1.Add(self.m_staticText2, 0, wx.ALL, 5)

		self.m_textCtrl1 = wx.TextCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0)
		fgSizer1.Add(self.m_textCtrl1, 0, wx.ALL, 5)

		self.m_button1 = wx.Button(self, wx.ID_ANY, u"Chosefile", wx.DefaultPosition, wx.DefaultSize, 0)
		fgSizer1.Add(self.m_button1, 0, wx.ALL, 5)

		self.m_button2 = wx.Button(self, wx.ID_ANY, u"run", wx.DefaultPosition, wx.DefaultSize, 0)
		fgSizer1.Add(self.m_button2, 0, wx.ALL, 5)

		bSizer1.Add(fgSizer1, 0, wx.EXPAND, 5)

		self.m_textCtrl2 = wx.TextCtrl(self, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize,
									   wx.TE_AUTO_URL | wx.TE_MULTILINE)
		bSizer1.Add(self.m_textCtrl2, 0, wx.ALL | wx.EXPAND, 5)

		self.m_panel1 = wx.Panel(self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL)
		bSizer1.Add(self.m_panel1, 1, wx.EXPAND | wx.ALL, 5)

		self.SetSizer(bSizer1)
		self.Layout()

		self.Centre(wx.BOTH)

		# Connect Events
		self.m_button1.Bind(wx.EVT_BUTTON, self.load)
		self.m_button2.Bind(wx.EVT_BUTTON, self.run)
		self.Bind(wx.EVT_CLOSE, self.close)
		self.filename = ""
		self.wildcard = "All files (*.*)|*.*|" \
						"Python source (*.py; *.pyc)|*.py;*.pyc"

	def __del__(self):
		pass

	# Virtual event handlers, overide them in your derived class
	def close(self,event):
		self.Parent.setbutton(2)
		self.Destroy()


	def load(self, event):
		dlg = wx.FileDialog(self, defaultFile="", wildcard=self.wildcard, style=wx.OPEN)
		if dlg.ShowModal() == wx.ID_OK:
			self.filename = dlg.GetPath()
			self.m_textCtrl1.SetValue(self.filename)

	def run(self, event):
		if  os.path.exists(self.filename) == False:
			wx.MessageBox("ERROR")
		else:
			dataset = loaddata(self.filename)
			attname = dataset[0]
			del dataset[0]
			print(dataset)
			del dataset[0]
			attname2 = attname[:]
			tree = createTree(dataset,attname)
			filename = self.filename.split(".")[0]
			createPlot(tree,filename + "res.png")
			sdata = []
			save(tree,sdata,0)
			f = open(filename + "res.txt",'w')
			f.writelines(sdata)
			f.close()
			#save(tree)



