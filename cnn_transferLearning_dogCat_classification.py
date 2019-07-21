import os
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3  #使用Inception网络

local_weights_file='/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5' #权重文件
pre_trained_model=InceptionV3(input_shape=(150,150,3),include_top=False,weights=None) #设置输入维度，最后一层不使用下载的权重
pre_trained_model.load_weights(local_weights_file) #载入权重
for layer in pre_trained_model.layers:  #锁定输出层之前层权重
    layer.trainable=False

pre_trained_model.summary() #打印摘要


#找到mixed7层并输出
last_layer=pre_trained_model.get_layer('mixed7')
print('last layer output shape:',last_layer.output_shape)
last_output=last_layer.output

from tensorflow.keras.optimizers import RMSprop #使用RMSprop优化算法

# 将之前mixed7层输出降到一维
x=layers.Flatten()(last_output)
#全连接层，1024个单元，使用Relu激活函数
x=layers.Dense(1024,activation='Relu')(x)
#使用dropout来抑制过拟合
x=layers.Dropout(0.2)(x)
#输出层
x=layers.Dense(1,activation='sigmoid')(x)

model=Model(pre_trained_model.input,x) #创建模型

model.compile(optimizer=RMSprop(lr=0.0001),loss='binary_crossentropy',metrics=['acc']) #设置优化算法，损失函数，评价指标



from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import zipfile

local_zip='//tmp/cats_and_dogs_filtered.zip'
zip_ref=zipfile.zipFile(local_zip,'r')
zip_ref.extractall('tmp')
zip_ref.close()

#数据文件
base_dir='/tmp/cats_and_dogs_filtered'

train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')

train_cats_dir=os.path.join(train_dir,'cats')#训练集-猫目录
train_dogs_dir=os.path.join(train_dir,'dogs')#训练集-狗目录
validation_cats_dir=os.path.join(validation_dir,'cats')#开发集-猫目录
validation_dogs_dir=os.path.join(validation_dir,'dogs')#开发集-狗目录

train_cat_fnames=os.listdir(train_cats_dir)
train_dog_fnames=os.listdir(train_dogs_dir)

#训练集使用数据增强，设置参数
train_datagen=ImageDataGenerator(rescale=1./255.,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

#开发集不使用数据增强
test_datagen=ImageDataGenerator(rescale=1.0/255.)

# mini-batch梯度下降
train_generator=train_datagen.flow_from_directory(train_dir,batch_size=20,class_mode='binary',target_size=(150,150))


validation_generator=test_datagen.flow_from_directory(validation_dir,batch_size=20,class_mode='binary',target_size=(150,150))

history=model.fit_generator(train_generator,validation_data=validation_generator,steps_per_epoch=100,epochs=20,validation_steps=50,verbose=2) #训练

import matplotlib.pyplot as plt
#画出训练集和开发集的准确率随迭代次数变化曲线
acc=history.history['acc']
val_acc=history.history['val_acc']
lss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs,acc,'r',label='Training accuracy')
plt.plot(epochs,cal_acc,'b',label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


