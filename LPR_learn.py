#导入需要的包

import numpy as np
import paddle as paddle
import paddle.fluid as fluid
# paddle.fluid.install_check.run_check()

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from multiprocessing import cpu_count
from paddle.fluid.dygraph import Pool2D,Conv2D
# from paddle.fluid.dygraph import FC
from paddle.fluid.dygraph import Linear


# 生成车牌字符图像列表
data_path = 'data/characterData/data'
character_folders = os.listdir(data_path)
label = 0
LABEL_temp = {}
if(os.path.exists('./train_data.list')):
    os.remove('./train_data.list')
if(os.path.exists('./test_data.list')):
    os.remove('./test_data.list')

for character_folder in character_folders:
    character_imgs = os.listdir(os.path.join(data_path, character_folder))
    for i in range(len(character_imgs)):
        mark = False
        for im in character_imgs[i]:
            if u'\u4e00' <= im <= u'\u9fff':
                mark = True
                break
        if mark:
            file_name = character_folder + '-' + '-'.join(character_imgs[i].split('_')[1:])
            oldfile_name = os.path.join(os.path.join(data_path, character_folder), character_imgs[i])
            newfile_name = os.path.join(os.path.join(data_path, character_folder), file_name)
            print(oldfile_name, newfile_name)
            os.rename(oldfile_name, newfile_name)

for character_folder in character_folders:
    with open('./train_data.list', 'a') as f_train:
        with open('./test_data.list', 'a') as f_test:
            if character_folder == '.DS_Store' or character_folder == '.ipynb_checkpoints' or character_folder == 'data23617':
                continue
            print(character_folder + " " + str(label))
            LABEL_temp[str(label)] = character_folder #存储一下标签的对应关系
            character_imgs = os.listdir(os.path.join(data_path, character_folder))
            for i in range(len(character_imgs)):
                if i%10 == 0:
                    f_test.write(os.path.join(os.path.join(data_path, character_folder), character_imgs[i]) + "\t" + str(label) + '\n')
                else:
                    f_train.write(os.path.join(os.path.join(data_path, character_folder), character_imgs[i]) + "\t" + str(label) + '\n')
    label = label + 1
print('图像列表已生成')



# 用上一步生成的图像列表定义车牌字符训练集和测试集的reader
def data_mapper(sample):
    img, label = sample
    img = paddle.dataset.image.load_image(file=img, is_color=False)
    # print(type(img))
    # img = img.flatten().astype('float32') / 255.0
    # img /= 255.0
    img = np.array(img).astype('float32')
    # img = img.transpose((2, 0, 1))
    img = img / 255.0
    return img, label

# def data_mapper(sample):
# 	img, label = sample
# 	img = Image.open(img)
# 	img = img.resize((100, 100), Image.ANTIALIAS)
# 	img = np.array(img).astype('float32')
# 	img = img.transpose((2, 0, 1))
# 	img = img / 255.0
# 	return img, label

def data_reader(data_list_path):
    def reader():
        with open(data_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)
    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)

# 用于训练的数据提供器
train_reader = paddle.batch(reader=paddle.reader.shuffle(reader=data_reader('./train_data.list'), buf_size=512), batch_size=128)
# 用于测试的数据提供器
test_reader = paddle.batch(reader=data_reader('./test_data.list'), batch_size=128)

#定义网络
class MyLeNet(fluid.dygraph.Layer):
    def __init__(self):
        super(MyLeNet,self).__init__()
        self.hidden1_1 = Conv2D(1, 28 , 5 , 1)
        self.hidden1_2 = Pool2D(pool_size=2, pool_type='max', pool_stride=1)
        self.hidden2_1 = Conv2D(28, 32, 3, 1)
        self.hidden2_2 = Pool2D(pool_size=2, pool_type='max', pool_stride=1)
        self.hidden3 = Conv2D(32, 32, 3, 1)
        self.hidden4 = Linear(32*10*10, 65, act='softmax')
    def forward(self,input):
        x = self.hidden1_1(input)
        x = self.hidden1_2(x)
        x = self.hidden2_1(x)
        x = self.hidden2_2(x)
        x = self.hidden3(x)
        x = fluid.layers.dropout(x, dropout_prob=0.1)
        x = fluid.layers.reshape(x, shape=[-1, 32*10*10])
        y = self.hidden4(x)
        return y


# with fluid.dygraph.guard():
with fluid.dygraph.guard(place = fluid.CUDAPlace(0)):
# with fluid.dygraph.guard(place=fluid.CPUPlace()):

	model = MyLeNet()  # 模型实例化
	model.train()  # 训练模式
	opt = fluid.optimizer.SGDOptimizer(learning_rate=0.001,
	                                   parameter_list=model.parameters())  # 优化器选用SGD随机梯度下降，学习率为0.001.

	# opt = fluid.optimizer.
#
# (learning_rate=0.001, momentum=0.9,
	#                                    parameter_list=model.parameters())  # Momentum，学习率为0.001.
	epochs_num = 680  # 迭代次数为2

	for pass_num in range(epochs_num):

		for batch_id, data in enumerate(train_reader()):
			images = np.array([x[0].reshape(1, 20, 20) for x in data], np.float32)
			labels = np.array([x[1] for x in data]).astype('int64')
			labels = labels[:, np.newaxis]
			image = fluid.dygraph.to_variable(images)
			label = fluid.dygraph.to_variable(labels)

			predict = model(image)  # 预测

			loss = fluid.layers.cross_entropy(predict, label)
			avg_loss = fluid.layers.mean(loss)  # 获取loss值

			acc = fluid.layers.accuracy(predict, label)  # 计算精度

			if batch_id != 0 and batch_id % 50 == 0:
				print(
					"train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num, batch_id, avg_loss.numpy(),
					                                                              acc.numpy()))

			avg_loss.backward()
			opt.minimize(avg_loss)
			model.clear_gradients()

	fluid.save_dygraph(model.state_dict(), 'MyLeNet')  # 保存模型

# 模型校验
with fluid.dygraph.guard():
	accs = []
	model = MyLeNet()  # 模型实例化
	model_dict, _ = fluid.load_dygraph('MyLeNet')
	model.load_dict(model_dict)  # 加载模型参数
	model.eval()  # 评估模式
	for batch_id, data in enumerate(test_reader()):  # 测试集
		images = np.array([x[0].reshape(1, 20, 20) for x in data], np.float32)
		labels = np.array([x[1] for x in data]).astype('int64')
		labels = labels[:, np.newaxis]

		image = fluid.dygraph.to_variable(images)
		label = fluid.dygraph.to_variable(labels)

		predict = model(image)  # 预测
		acc = fluid.layers.accuracy(predict, label)
		accs.append(acc.numpy()[0])
		avg_acc = np.mean(accs)
	print(avg_acc)

# 对车牌图片进行处理，分割出车牌中的每一个字符并保存
license_plate = cv2.imread('data/chepai.png')
gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_RGB2GRAY)
ret, binary_plate = cv2.threshold(gray_plate, 175, 255, cv2.THRESH_BINARY)
result = []
for col in range(binary_plate.shape[1]):
	result.append(0)
	for row in range(binary_plate.shape[0]):
		result[col] = result[col] + binary_plate[row][col] / 255
character_dict = {}
num = 0
i = 0
while i < len(result):
	if result[i] == 0:
		i += 1
	else:
		index = i + 1
		while result[index] != 0:
			index += 1
		character_dict[num] = [i, index - 1]
		num += 1
		i = index

for i in range(8):
	if i == 2:
		continue
	padding = (170 - (character_dict[i][1] - character_dict[i][0])) / 2
	ndarray = np.pad(binary_plate[:, character_dict[i][0]:character_dict[i][1]], ((0, 0), (int(padding), int(padding))),
	                 'constant', constant_values=(0, 0))
	ndarray = cv2.resize(ndarray, (20, 20))
	cv2.imwrite('./' + str(i) + '.png', ndarray)


def load_image(path):
	img = paddle.dataset.image.load_image(file=path, is_color=False)
	img = img.astype('float32')
	img = img[np.newaxis,] / 255.0
	return img


#将标签进行转换
print('Label:',LABEL_temp)
match = {'A':'A','B':'B','C':'C','D':'D','E':'E','F':'F','G':'G','H':'H','I':'I','J':'J','K':'K','L':'L','M':'M','N':'N',
        'O':'O','P':'P','Q':'Q','R':'R','S':'S','T':'T','U':'U','V':'V','W':'W','X':'X','Y':'Y','Z':'Z',
        'yun':'云','cuan':'川','hei':'黑','zhe':'浙','ning':'宁','jin':'津','gan':'赣','hu':'沪','liao':'辽','jl':'吉','qing':'青','zang':'藏',
        'e1':'鄂','meng':'蒙','gan1':'甘','qiong':'琼','shan':'陕','min':'闽','su':'苏','xin':'新','wan':'皖','jing':'京','xiang':'湘','gui':'贵',
        'yu1':'渝','yu':'豫','ji':'冀','yue':'粤','gui1':'桂','sx':'晋','lu':'鲁',
        '0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9'}
L = 0
LABEL ={}

for V in LABEL_temp.values():
    LABEL[str(L)] = match[V]
    L += 1
print(LABEL)


#构建预测动态图过程
with fluid.dygraph.guard():
    model=MyLeNet()#模型实例化
    model_dict,_=fluid.load_dygraph('MyLeNet')
    model.load_dict(model_dict)#加载模型参数
    model.eval()#评估模式
    lab=[]
    for i in range(8):
        if i==2:
            continue
        infer_imgs = []
        infer_imgs.append(load_image('./' + str(i) + '.png'))
        infer_imgs = np.array(infer_imgs)
        infer_imgs = fluid.dygraph.to_variable(infer_imgs)
        result=model(infer_imgs)
        lab.append(np.argmax(result.numpy()))
# print(lab)


# display(Image.open('./车牌.png'))
print('\n车牌识别结果为：',end='')
for i in range(len(lab)):
    print(LABEL[str(lab[i])],end='')


# save record


