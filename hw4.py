# William Dahl	
# 001273655	
# ICSI 431 Data Mining

#imports required modules
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import kde
import pandas as pd
from matplotlib import ticker
from wordcloud import WordCloud
from statsmodels import robust

#creates a histogram of the sexs of Albalone
def Histogram():
	sex = []#holds the sex's in the data file
	#opens the data file as f
	with open("data.txt", "r") as f:
		#loops throught the data file
		for i in range(0, 4176):
			line = f.readline()#reads the line
			sex.append(line[0])#appends the sex to the sex list

		num_bins = 3 #numbers of bins to be used in the histograme
		#creates the histogram
		n, bins, patches = plt.hist(sex, num_bins, facecolor='blue', alpha=0.5)

		plt.xlabel('Sex')
		plt.ylabel('Frecuency')
		plt.title("Distrobution of Males, Females, and Infants collected")
		plt.show() #displays the histogram
 
#Scater plot of the diffrent sexs and there relation between length and diameter
def Scatter_plot():
	tokens = [] #holds the tokens of th eline when split
	mx = [] #holds male lengths
	my =[]#male diameter
	fx = []#female length
	fy = []#female diameter
	ix = []#infant length
	iy = []#infant diameter
	with open("data.txt", "r") as f:
		for i in range(0, 4176):
			line = f.readline()
			tokens.append(line.split(","))#splits the line at commas and puts each token into tokens list

	#loops through tokens array
	for i in range(0,4176):
		#checks for the sex of the albalone and puts data into the coresponding list
		if tokens[i][0] == 'M':
			mx.append(tokens[i][1])#adds the length to the list
			my.append(tokens[i][2])#adds the diameter to the list

		if tokens[i][0] == 'F':
			fx.append(tokens[i][1])
			fy.append(tokens[i][3])

		if tokens[i][0] == 'I':
			ix.append(tokens[i][1])
			iy.append(tokens[i][2])


	#creates the scatter plot
	plt.scatter(mx, my, color='red', s=0.001, label='Male', marker='*')
	plt.scatter(fx, fy, color='blue', s=0.001, label = 'Female', marker='*')
	plt.scatter(ix, iy, color='green', s=0.001, label = 'Infant', marker='*')
	plt.title('Length vs Diameter')
	plt.xlabel('length')
	plt.ylabel('Diameter')
	plt.legend(loc=2)
	plt.show()

#box plot of the height of the abalone
def Box_plot():
	height = [] #holds height of abaolnes
	with open("data.txt", "r") as f:
		for i in range(0, 4176):
			line = f.readline().split(",")
			height.append(float(line[3]))#changes the data to a float type


	plt.boxplot(height)
	plt.xlabel("Height")
	plt.title("Height of abalone")
	plt.show()

#density map of total weight and meat weight
def Density_map():
	total_weight = []#holds total weight
	meat_weight = []#holds meat weight
	with open("data.txt", "r") as f:
		for i in range(0, 4176):
			line = f.readline().split(",")
			total_weight.append(float(line[4]))
			meat_weight.append(float(line[5]))

	nbins=300#numbers of bins to organize data
	#creates a guassian of the data
	k = kde.gaussian_kde([total_weight,meat_weight])
	#makes an mgrid out of the min and max of the two weigths
	xi, yi = np.mgrid[min(total_weight):max(total_weight):nbins*1j, min(meat_weight):max(meat_weight):nbins*1j]
	#flattens the mgrids
	zi = k(np.vstack([xi.flatten(), yi.flatten()]))
	#makes the density map
	plt.title("Distrobution of Total Weight and Weight of Meat of Abalone")
	plt.xlabel("Total Weight")
	plt.ylabel("Meat weight")
	plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
	plt.show()

#parrallel coordinates of diffrent weight datet based on sex
def Parallel_coordinate():
	df = pd.read_csv('data.csv')#reads a csv file and sets it to df
	plt.figure()#creates a figure
	#plotts corrdinates based on the data catigories in the csv file
	pd.plotting.parallel_coordinates(df[['Sex', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']], 'Sex')
	#creates the parrallel coordinates plot
	plt.title("Weights of abalone by Sex")
	plt.ylabel("Weight in grams")
	plt.xlabel("catigories")
	plt.show()
 
#correlation matrix to show correlation between the diffrent weights in the data
def Correlation_matrix():
	#intialized matrix
	w, h = 4, 4; 
	Matrix = [[0 for x in range(w)] for y in range(h)] 

	with open("data.txt", "r") as f:
		for i in range(0, 4):
			line = f.readline().split(",")
			for j in range(4, 7):
				Matrix[i][j-4] = float(line[j])

	#creates the matrix correlation plot
	plt.matshow(Matrix)

	#creates the groups in the corrlation matrix
	groups = ['Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']

	#creates the titles lables for the axises
	x_pos = np.arange(len(groups))
	plt.xticks(x_pos, groups)

	y_pos = np.arange(len(groups))
	plt.yticks(y_pos, groups)

	#displays the matrix
	plt.title("Correlation between diffrent weights of the Abalone")	
	plt.show()

#Creates a word clound based on randomly sampled tweets
def Word_cloud():
	text = open("random_data.txt", "r").read()#reads file and puts into text
	wordcloud = WordCloud(max_font_size=40).generate(text)#generates a wordcloud out of text with max font size of 40
	#creates the word cloud
	plt.figure()
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")#turns off the axises for the wordcloud
	plt.title("Word Cloud made from randomly sampled tweets")
	plt.show()

#pritns the summary of the statistcs for every catigory from the data file
def Summary_of_statistics():
	df = pd.read_csv('data.csv')
	print "Length"
	print "--------------------"
	print "Standard deviation: ", np.std(df['Length'])
	print "Mean: ", np.mean(df['Length'])
	print "Median: ", np.median(df['Length'])
	print "MAD: ", robust.mad(df['Length'])
	print "Max: ", np.max(df['Length'])
	print "Min: ", np.min(df['Length'])
	print "\n"
	print "Height"
	print "--------------------"
	print "Standard deviation: ", np.std(df['Height'])
	print "Mean: ", np.mean(df['Height'])
	print "Median: ", np.median(df['Height'])
	print "MAD: ", robust.mad(df['Height'])
	print "Max: ", np.max(df['Height'])
	print "Min: ", np.min(df['Height'])
	print "\n"
	print "Diameter"
	print "--------------------"
	print "Standard deviation: ", np.std(df['Diameter'])
	print "Mean: ", np.mean(df['Diameter'])
	print "Median: ", np.median(df['Diameter'])
	print "MAD: ", robust.mad(df['Diameter'])
	print "Max: ", np.max(df['Diameter'])
	print "Min: ", np.min(df['Diameter'])
	print "\n"
	print "Whole weight"
	print "--------------------"
	print "Standard deviation: ", np.std(df['Whole weight'])
	print "Mean: ", np.mean(df['Whole weight'])
	print "Median: ", np.median(df['Whole weight'])
	print "MAD: ", robust.mad(df['Whole weight'])
	print "Max: ", np.max(df['Whole weight'])
	print "Min: ", np.min(df['Whole weight'])
	print "\n"
	print "Shucked weight"
	print "--------------------"
	print "Standard deviation: ", np.std(df['Shucked weight'])
	print "Mean: ", np.mean(df['Shucked weight'])
	print "Median: ", np.median(df['Shucked weight'])
	print "MAD: ", robust.mad(df['Shucked weight'])
	print "Max: ", np.max(df['Shucked weight'])
	print "Min: ", np.min(df['Shucked weight'])
	print "\n"
	print "Viscera weight"
	print "--------------------"
	print "Standard deviation: ", np.std(df['Viscera weight'])
	print "Mean: ", np.mean(df['Viscera weight'])
	print "Median: ", np.median(df['Viscera weight'])
	print "MAD: ", robust.mad(df['Viscera weight'])
	print "Max: ", np.max(df['Viscera weight'])
	print "Min: ", np.min(df['Viscera weight'])
	print "\n"
	print "Shell weight"
	print "--------------------"
	print "Standard deviation: ", np.std(df['Shell weight'])
	print "Mean: ", np.mean(df['Shell weight'])
	print "Median: ", np.median(df['Shell weight'])
	print "MAD: ", robust.mad(df['Shell weight'])
	print "Max: ", np.max(df['Shell weight'])
	print "Min: ", np.min(df['Shell weight'])
	print "\n"
	print "Rings"
	print "--------------------"
	print "Standard deviation: ", np.std(df['Rings'])
	print "Mean: ", np.mean(df['Rings'])
	print "Median: ", np.median(df['Rings'])
	print "MAD: ", robust.mad(df['Rings'])
	print "Max: ", np.max(df['Rings'])
	print "Min: ", np.min(df['Rings'])

if __name__ == '__main__':
	Histogram()
	Scatter_plot()
	Box_plot()
	Density_map()
	Parallel_coordinate()
	Correlation_matrix()
	Word_cloud()
	Summary_of_statistics()