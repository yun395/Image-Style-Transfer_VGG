import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from copy import deepcopy
import matplotlib.pyplot as plt
from torchvision.models import VGG19_Weights  # 新增这行


#10.4.2 计算Gram矩阵函数的实现
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G

#10.3.2 内容损失模块的实现
class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = torch.sum((input-self.target) ** 2) / 2.0
        return input


#10.4.3 风格损失模块的实现
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        a, b, c, d = input.size()
        G = gram_matrix(input)
        self.loss = torch.sum((G-self.target) ** 2) / (4.0 * b * b * c * d)
        return input


#10.6.1 图像预处理
class ImageCoder:
    def __init__(self, image_size, device):

        self.device = device

        self.preproc = transforms.Compose([
            transforms.Resize(image_size),  # 改变图像大小
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化
                                 std=[1, 1, 1]),
            transforms.Lambda(lambda x: x.mul_(255))
        ])

        self.postproc = transforms.Compose([
            transforms.Lambda(lambda x: x.mul_(1./255)),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1,1,1])
        ])

        self.to_image = transforms.ToPILImage()

    def encode(self, image_path):
         # 修改点1：添加.convert('RGB')，强制转为3通道RGB
        image = Image.open(image_path).convert('RGB') 
       # image = Image.open(image_path)
        image = self.preproc(image)
        image = image.unsqueeze(0)
        return image.to(self.device, torch.float)

    def decode(self, image):
        image = image.cpu().clone()
        image = image.squeeze()
        image = self.postproc(image)
        image = image.clamp(0, 1)
        return self.to_image(image)


#10.6.2 参数定义
content_layers = ['conv_4_2'] # 内容损失函数使用的卷积层
style_layers = ['conv_1_1', 'conv2_1', 'conv_3_1', 'conv_4_1', 'conv5_1'] # 风格损失函数使用的卷积层
content_weights = [1] # 内容损失函数的权重
style_weights = [1e3, 1e3, 1e3, 1e3, 1e3] # 风格损失函数的权重
num_steps=200 # 最优化的步数


#模型初始化
class Model:

    def __init__(self, device, image_size):

        #cnn = torchvision.models.vgg19(pretrained=True).features.to(device).eval()
        cnn = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        #vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.cnn = deepcopy(cnn) # 获取预训练的VGG19卷积神经网络
        self.device = device

        self.content_losses = []
        self.style_losses = []

        self.image_proc = ImageCoder(image_size, device)


#10.6.4 运用风格迁移的主函数
    def run(self, content_image_path, style_image_path):

        content_image = self.image_proc.encode(content_image_path)
        style_image = self.image_proc.encode(style_image_path)

        self._build(content_image, style_image) # 建立损失函数
        output_image = self._transfer(content_image) # 进行最优化

        return self.image_proc.decode(output_image)


#10.6.5 利用VGG网络建立损失函数
    def _build(self, content_image, style_image):

        self.model = nn.Sequential()

        block_idx = 1
        conv_idx = 1

        # 逐层遍历VGG19，取用需要的卷积层
        for layer in self.cnn.children():

            # 识别该层类型并进行编号命名
            if isinstance(layer, nn.Conv2d):
                name = 'conv_{}_{}'.format(block_idx, conv_idx)
                conv_idx += 1
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}_{}'.format(block_idx, conv_idx)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(block_idx)
                block_idx += 1
                conv_idx = 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(block_idx)
            else:
                raise Exception("invalid layer")

            self.model.add_module(name, layer)

            # 添加内容损失函数
            if name in content_layers:
                target = self.model(content_image).detach()
                content_loss = ContentLoss(target)
                self.model.add_module("content_loss_{}_{}".format(block_idx, conv_idx), content_loss)
                self.content_losses.append(content_loss)

            # 添加风格损失函数
            if name in style_layers:
                target_feature = self.model(style_image).detach()
                style_loss = StyleLoss(target_feature)
                self.model.add_module("style_loss_{}_{}".format(block_idx, conv_idx), style_loss)
                self.style_losses.append(style_loss)

        # 留下有用的部分
        i = 0
        for i in range(len(self.model) - 1, -1, -1):
            if isinstance(self.model[i], ContentLoss) or isinstance(self.model[i], StyleLoss):
                break
        self.model = self.model[:(i + 1)]


#10.6.6 风格迁移的优化过程
    def _transfer(self, content_image):

        output_image = content_image.clone()
        random_image = torch.randn(content_image.data.size(), device=self.device)
        output_image = 0.4 * output_image + 0.6 * random_image

        optimizer = torch.optim.LBFGS([output_image.requires_grad_()])

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():

                optimizer.zero_grad()
                self.model(output_image)
                style_score = 0
                content_score = 0

                for sl, sw in zip(self.style_losses, style_weights):
                    style_score += sl.loss * sw
                for cl, cw in zip(self.content_losses, content_weights):
                    content_score += cl.loss * cw

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("iteration {}: Loss: {:4f} Style Loss: {:4f} Content Loss: {:4f}"
                          .format(run, loss.item(), style_score.item(), content_score.item()))
                return loss

            optimizer.step(closure)

        return output_image



#运用风格迁移
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 256
model = Model(device, image_size)

style_image_path = 'E:/PyTorch/PyTorch/10/van_gogh.png'
content_image_path = 'E:/PyTorch/PyTorch/10/street.png'
out_image = model.run(content_image_path, style_image_path)
plt.imshow(out_image)
plt.show()



