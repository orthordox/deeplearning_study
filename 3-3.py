import torch
import torch.optim as optim

x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  80],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# 가중치와 편향 선언
W = torch.tensor([[1.2561],
                  [0.6549],
                  [0.1360]], requires_grad=True)
b = torch.tensor([-2.2868], requires_grad = True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 100000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드 캐스팅되어 각 샘플에 더해집니다.
    hypothesis = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch %10000 == 0:
        print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f} W: {} b: {}'.format(
            epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item(), W.squeeze(), b
        ))