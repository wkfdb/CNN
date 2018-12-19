# encoding=utf-8
import tensorflow as tf
#input_tensor--输入层
#train--
#regularizer--
#reuse--当reuse为False或者None时（这也是默认值），同一个tf.variable_scope下面的变量名不能相同；
#-------当reuse为True时，tf.variable_scope只能获取已经创建过的变量。
#-------reuse=False时，tf.variable_scope创建变量；reuse=True时，tf.variable_scope获取变量。

#tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
#---第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
#具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，类型为float32或float64
#---第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]
#这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
#---第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4？？为什么是4维？
#---第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
#SAME表示卷积后的长宽=图像的长宽，VALID表示卷积后的维度根据图片大小核卷积核维数来确定
#---第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
#---结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。

#pool汇聚层
#在实践中，最大汇聚层通常只有两种形式：一种是F=3,S=2，也叫重叠汇聚（overlapping pooling），
#另一个更常用的是F=2,S=2。对更大感受野进行汇聚需要的汇聚尺寸也更大，而且往往对网络有破坏性。
#tf.nn.max_pool(value, ksize, strides, padding, name=None)
#---value，同卷积层的input
#---ksize, 滤波器的大小，因为不会对batch和channel进行汇聚，所以一般为[1, height, width, 1]
#---strides,在每个维度上移动的步长，一般也是[1, stride,stride, 1]
#---padding同卷积层的padding
def infer(input_tensor, train, regularizer, reuse):
	#"layer1-conv1"第一层卷积层，scope-name
    with tf.variable_scope("layer1-conv1",reuse=reuse):
    	#在第一层内
    	#创建变量，3*3*3的卷积核，共有16个卷积核，初始化为均值为0标准差为0.1的正态分布
        conv1_weights=tf.get_variable("weight",[3,3,3,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #创建偏移，一维向量，运算的时候会自动进行多维展开，初始化为常数0
        conv1_biases=tf.get_variable("bias",[16],initializer=tf.constant_initializer(0.0))
        #进行卷积，步长为1，每个filter卷积出来的feature map都和原来图像一样大，产生16个feature map
        conv1=tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        #激活过程
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    # with tf.variable_scope("layer2-pool1",reuse=reuse):
    #     pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #第二个卷积层
    with tf.variable_scope("layer3-conv2",reuse=reuse):
    	#3*3*16的卷积核，共有16个，初始化为均值为0标准差为0.1的正态分布
        conv2_weights=tf.get_variable("weight",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #偏差
        conv2_biases=tf.get_variable("bias",[16],initializer=tf.constant_initializer(0.0))
        #再次卷积，产生的feature map集合的维数和第一层的结果相同
        conv2=tf.nn.conv2d(relu1,conv2_weights,strides=[1,1,1,1],padding='SAME')
        #激活
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    #第三层池化层--pool 也叫汇聚层，拿来降维
    with tf.variable_scope("layer4-pool2",reuse=reuse):
    	#降维，维度在长宽方向上将为原来的一半
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #第四层--卷积层
    with tf.variable_scope("layer5-conv3",reuse=reuse):
    	#32个卷积核，输出的深度变大
        conv3_weights=tf.get_variable("weight",[3,3,16,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases=tf.get_variable("bias",[32],initializer=tf.constant_initializer(0.0))
        conv3=tf.nn.conv2d(pool2,conv3_weights,strides=[1,1,1,1],padding="SAME")
        relu3=tf.nn.relu(tf.nn.bias_add(conv3,conv3_biases))
    #第五层--卷积层
    with tf.variable_scope("layer6-conv4",reuse=reuse):
        conv4_weights = tf.get_variable("weight",[3,3,32,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias",[32],initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(relu3,conv4_weights,strides=[1,1,1,1],padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4,conv4_biases))
    #第六层--汇聚层
    with tf.variable_scope("layer7-pool3",reuse=reuse):
        pool3=tf.nn.max_pool(relu4,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # 转换为输出向量
    #x.get_shape()，只有tensor才可以使用这种方法，返回的是一个元组。需要通过as_list()的操作转换成list.
    pool3_shape=pool3.get_shape().as_list()
    nodes=pool3_shape[1]*pool3_shape[2]*pool3_shape[3]
    #tf.reshape(tensor,shape,name=None) 一个batch有多张图像，pool3_shape[0]就是每个batch的图象数，剩下的结果转换为一维向量
    reshaped=tf.reshape(pool3,[pool3_shape[0],nodes])

    #开始全连接
    with tf.variable_scope("layer8-fc1",reuse=reuse):
    	#创建权重，转换为64维数据
        fc1_weights=tf.get_variable("weight",[nodes,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
        	#tf.add_to_collection：把变量放入一个集合，把很多变量变成一个列表，即将tf变量加入到一个列表里，loss是列表名
			#tf.get_collection：从一个结合中取出全部变量，是一个列表
            tf.add_to_collection("loss",regularizer(fc1_weights))
        fc1_biases=tf.get_variable("bias",[64],initializer=tf.constant_initializer(0.1))
        #不再是二维卷积而是矩阵乘法运算
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        # if train:
        #     fc1=tf.nn.dropout(fc1,keep_prob=0.9)
    #全连接层2
    with tf.variable_scope("layer9-fc2",reuse=reuse):
    	#降维
        fc2_weights=tf.get_variable("weight",[64,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("loss",regularizer(fc2_weights))
        fc2_biases=tf.get_variable("bias",[32],initializer=tf.constant_initializer(0.1))
        fc2=tf.nn.relu(tf.matmul(fc1,fc2_weights)+fc2_biases)
        # if train:
        #     fc2=tf.nn.dropout(fc2,keep_prob=0.9)
    #全连接层3
    with tf.variable_scope("layer10-fc3",reuse=reuse):
        fc3_weight=tf.get_variable("weight",[32,12],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("loss",regularizer(fc3_weight))
        fc3_biases=tf.get_variable("bias",[12],initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fc2,fc3_weight)+fc3_biases
    return tf.nn.softmax(logit)

