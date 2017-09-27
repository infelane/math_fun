import numpy as np

import tkinter as tk

















class Player(object):

	def __init__(self, name, val, color = 'pink'):
		self.name = name
		self.val = val
		self.color = color

class Board(object):

	counter = 0

	def __init__(self, player1, player2, x = 19, y = 19):
		self.values = np.zeros((x, y))
		self.colors = [ [None]*19 for _ in range(19) ]
		self.player1 = player1
		self.player2 = player2


	def move(self, player, x, y):
		if self.values[x,y] == 0:
			self.values[x, y] = player.val
			self.colors[x][y] = player.color

		else:
			print("Already played here")


	def show(self):
		print("TODO")
		root = tk.Tk()

		# counter = 0 #Keeps track of iteration and


		def on_click(i, j, event):
			# global counter
			color = self.player1.color if self.counter % 2 else self.player2.color
			# event.widget.config(bg=color)

			event.widget.config(bg=color)

			player = self.player1 if self.counter % 2 else self.player2
			self.move(player, i,j)


			self.counter += 1




		for i, row in enumerate(self.values):
			for j, column in enumerate(row):
				L = tk.Label(root, text='    ', bg='grey')
				L.grid(row=i, column=j)
				L.bind('<Button-1>', lambda e, i=i, j=j: on_click(i, j, e))
		root.mainloop()




def main():
	white = Player("white", -1, color = "white")
	black = Player("black",  1, color = "black")

	board = Board(white, black)

	x = np.random.randint(0, 18)
	y = np.random.randint(0, 18)

	board.move(white, x, y)

	board.show()



if __name__ == '__main__':
	main()