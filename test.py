import  torch


if __name__ == '__main__':
	out1 = torch.load('results/train.txt')
	out2 = torch.load('results/test.txt')

	epsilon = 1e-5
	print((out1 - out2) < epsilon)
	print(out1)
	print(out2)

	res = ((out1 - out2) > epsilon).float().sum().item()
	if res == 0:
		print('Test succeed')
	else:
		print("Test failed")