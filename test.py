import  torch


if __name__ == '__main__':
	out1 = torch.load('results/train.txt')
	out2 = torch.load('results/test.txt')

	print((out1 - out2) < 1e-5)
	print(out1)
	print(out2)