# region 加载库,基础参数配置
# 运行前下载数据集
# wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
# tar xvf simple-examples.tgz
# 下载PTB，借助reader读取数据内容，将单词转为唯一的数字编码
# git clone https://github.com/tensorflow/models.git
# 如果代码运行时出现TypeError: a bytes-like object is required, not 'str'
# 则将models/tutorials/rnn/ptb/reader.py中的return f.read().replace("\n", "<eos>").split() 改成f.read().decode("utf-8").replace("\n", "<eos>")

import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
# 把解压后的ptb路径添加进系统路径，这样这样系统才能找到并载入reader
import sys
# 本程序文件能够运行，说明本文件夹正是系统路径之一，所以系统会把这个相对路径附加到本程序路径下。
sys.path.append('models/tutorials/rnn/ptb')
import reader
# endregion

# region 模型参数类,用于配置构建和执行计算图的参数
# 封装好参数，方便训练前选择配置。
# 小模型。本代码用到的模型参数。
class SmallConfig(object):
  init_scale = 0.1      # 网络中权重值的初始scale
  learning_rate = 1.0   # 学习速率的初始值
  max_grad_norm = 5     # 梯度的最大范数
  num_layers = 2        # LSTM可以堆叠的层数
  num_steps = 20        # LSTM梯度反向传播的展开步数
  hidden_size = 200     # LSTM内的隐含节点数
  max_epoch = 1#4         # 初始学习速率可训练的epoch数，在此之后需要调整学习速率
  max_max_epoch = 13    # 总共可以训练的epoch数
  keep_prob = 1.0       # 保留节点的比率
  lr_decay = 0.5        # 学习速率的衰减速度
  batch_size = 20       # 每个batch中样本的数量
  vocab_size = 10000

# 中型模型
class MediumConfig(object):
  init_scale = 0.05     # 减小了init_scale,即希望权重初值不要过大，小一些有利于温和的训练.被tf.random_uniform_initializer()调用
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35        # 从20增大到了35
  hidden_size = 650     # 伴随num_steps,也相应地增大了约3倍
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5       # 引入dropout
  lr_decay = 0.8        # 因为学习的迭代次数增大，所以学习速率的衰减速度减小了。
  batch_size = 20
  vocab_size = 10000

# 大型模型
class LargeConfig(object):
  init_scale = 0.04     # 进一步缩小init_scale
  learning_rate = 1.0
  max_grad_norm = 10    # 放宽范数到10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500    # 提升到1500
  max_epoch = 14        # 增大
  max_max_epoch = 55    # 增大
  keep_prob = 0.35      # 模型复杂度上升，keep_prob调小
  lr_decay = 1 / 1.15   # 学习速率的衰减速度进一步减小。
  batch_size = 20
  vocab_size = 10000

# 测试用，所有的参数都尽量使用最小值，为了测试可以完整运行模型
class TestConfig(object):
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

# 配置参数，供下文的tf.Graph()和sv.managed_session()中使用。
config = SmallConfig()       # 配置训练时（广义的训练，包括train和valid）的参数
eval_config = SmallConfig()  # 配置测试时（即test）的参数。除了以下两个参数设为1，其他的参数全部与训练时相同。
eval_config.batch_size = 1   # 一个batch中样本的数量
eval_config.num_steps = 1    # LSTM的展开步数（unrolled steps of LSTM）

# endregion

# region 0 原始数据读入
# 包含三类。用于下文的tf.Graph()中调用输入数据类PTBInput()初始化出三种数据对象
# 本程序文件能够运行，说明本文件夹正是系统路径之一，所以可以直接填相对路径
# 从raw_data中得到训练数据，验证数据，测试数据。
raw_data = reader.ptb_raw_data('simple-examples/data/')
train_data, valid_data, test_data, _ = raw_data
# endregion

# region 1 计算图构建类

# region 1.1 参数配置/数据装载类
# 定义输入数据类
# 将在tf.Graph()中用于实例化三个数据对象：train_input、valid_input、test_input
# 注意，每个数据对象均包含input＿data和targets
class PTBInput(object):
    def __init__(self, config, data, name=None):
        self.batch_size = batch_size = config.batch_size
        # LSTM的展开步数（unrolled steps of LSTM）
        self.num_steps = num_steps = config.num_steps
        # 每一个epoch内需要的多少轮迭代
        self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
        #  获取特征数据input_data，label数据targets，每次执行获取一个batch的数据。
        #
        # self.targets的shape是[batch_size,num_steps]
        self.input_data, self.targets = reader.ptb_producer(data, batch_size, num_steps, name=name)
# endregion

# region 1.2 模型类
# 定义语言模型类
# 将在tf.Graph()中被实例化了三次，生成三个模型对象：m、mvalid、mtest
# 例如m=PTBModel(is_training=True, config=config, input_=train_input)，
# 其中，train_input是PTBInput()类传入原始数据完成实例化后的对象。
#　包含３类方法:模型初始化方法__init__(),学习速率更新的执行方法assign_lr(),属性读取方法(6个)
class PTBModel(object):

    def __init__(self, is_training, config, input_):

        # region 配置参数
        self._input = input_              # self._input仅被用于这个类的属性返回方法．在这个初始化方法内部直接用input＿
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        # hidden_size是LSTM的节点数
        size = config.hidden_size
        # vocab_size是词汇表的大小
        vocab_size = config.vocab_size
        # endregion

        # region 输入数据向量化
        # 词嵌入embedding部分
        # 将one-hot格式的单词转化为向量形式
        with tf.device("/cpu:0"):
            # 初始化embedding矩阵，行数设为词汇表vocab_size，列数（即每个单词的向量表达的维数）设为hidden_size（即LSTM单元中的隐含节点数）。
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=tf.float32)
            # 查询单词对应的向量表达式inputs。
            # input_是在该类被外部实例化时传入的，因此不需要占位符
            # input_.input_data shape=[batch_size]
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
            # inputs的shape =[batch_size,embedding_size]
        # 如果处于训练状态，且keep_prob小于1，则再添一个Dropout层
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        # endregion

        # region 定义LSTM结构
        # 设置默认的LSTM单元个数。利用tf.contrib.rnn.BasicLSTMCell()
        def lstm_cell():
            #传入的size为是LSTM的节点数，forget_bias即为forget gate的bias，state_is_tuple=True表示接收和返回的state是2-tuple形式
            return rnn.BasicLSTMCell(
                size, forget_bias=0.0, state_is_tuple=True)
        # 如果是训练中,则这句没意义,因为之后的if语句会重新定义attn_cell()就成了一个后接dropout 的cell层
        # 如果不是训练中,则这句就起到承前启后,跳过if语句,使cell层顺利堆叠.
        attn_cell = lstm_cell
        # 如果处于训练状态，且keep_prob小于1，则在lstm_cell之后接一个Dropout层
        if is_training and config.keep_prob < 1:
            def attn_cell():
                return rnn.DropoutWrapper(lstm_cell(), output_keep_prob=config.keep_prob)
        # 设置LSTM的堆叠层数
        # 使用RNN堆叠函数将前面构造的lstm_cell多层堆叠成cell，堆叠次数为config.num_layers
        # 如果num_layers是2，则结果形如[attn_cell() attn_cell()]
        cell = rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        # 有了LSTM cell，初始化cell为0。
        # LSTM单元读入一个单词并结合之前储存的状态state计算下一个单词出现的概率分布，每次读取一个单词后它的状态state会被更新。
        # 经过初始化，这里的self._initial_state就表征cell的初始状态，会由m.inital_state拿出来供外部使用，也会在本类下文中拿来初始化state。
        # state的shape是[batch_size,size*num_layers]
        self._initial_state = cell.zero_state(batch_size, tf.float32)
        # endregion

        # region 定义前向计算和cost

        # 设置LSTM的时间序列的深度num_step.
        inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # state的shape是[batch_size,size*num_layers]
        # outputs的shape就是[batch_size,num_steps,size]
        outputs, state = rnn.static_rnn(cell, inputs,
                                   initial_state=self._initial_state)
        '''
        # 定义输出outputs
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            # 将接下来的操作名称设为RNN
            # 限制梯度在反向传播时可以展开的步数为一个固定值，即步数num_steps
            for time_step in range(num_steps):
                # 从第二次循环开始，设置复用变量
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # 每次循环内，传入inputs和state到堆叠的LSTM单元（即cell）中
                # inputs有三个维度，含义依次是：batch中第几个样本，样本中第几个单词，单词的向量表达的维度。
                # inputs[:, time_step, :]代表所有样本的第time_step个单词。
                # 得到输出cell_output和更新后的state。
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                # 将time_step这个时间点的cell_output(shape是[batch_size,size])添加到输出列表中。
                outputs.append(cell_output)
            ＃当循环结束，outputs是一个有num_steps个元素的列表，每个元素的shape就是[batch_size,size]
        '''

        # 将outputs的内容串联起来，并转换为一维向量。
        # tf.concat(outputs, 1)的shape就是[batch_size*num_steps,size]
        # [-1,size]就表示每行必须是第二维必须是size个元素，然后第一维度是多少就是多少．
        # 结果就是output的shape就是[batch_size*num_steps,size]，相当于不变．
        output = tf.reshape(tf.concat(outputs, 1), [-1, size])
        # softmax层
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        # 输出的logits的shape是[batch_size*num_steps,vocab_size]　
        logits = tf.matmul(output, softmax_w) + softmax_b
        # 损失函数，计算输出logits和targets的偏差
        # 这里的sequence_loss即target words的average negative log probability,
        # 其定义是loss=-(1/N)*[lnP(target_1)+lnP(target_2)+...+lnP(target_N)],其中P(target_N)是target N的概率。
        # tf.contrib.legacy_seq2seq.sequence_loss_by_example返回一个序列的交叉熵的和
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],# 预测值
            [tf.reshape(input_.targets, [-1])],
            # 把标签值的shape[batch_size,num_steps]二维数组压缩成一维数组
            [tf.ones([batch_size * num_steps], dtype=tf.float32)]
            # 损失的权重，在这里所有的权重都为1，也就是说不同batch和不同时刻的重要程度是一样的
        )
        # 汇总batch的误差，计算平均到每个样本的平均损失
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        # 保留最终的状态为final_state
        self._final_state = state
        # endregion

        # region 定义训练操作
        # 只有训练模型时，才会执行这部分
        if not is_training:
            return
        # 定义学习速率_lr
        # 并将其设为不可训练
        self._lr = tf.Variable(0.0, trainable=False)
        # 定义优化器SGD
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # 定义训练op
        # minimize分两步，先取compute_gradient,再用apply_gradient
        # 获取全部可训练参数tvars
        tvars = tf.trainable_variables()
        # 针对前面得到的cost，计算tvars的梯度，并设置梯度的最大范数max_grad_norm
        # 这就是Gradient Clipping方法，控制梯度的最大范数，某种程度上起到正则化的效果，可防止梯度爆炸。
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),config.max_grad_norm)
        #　apply_gradients()将前面clip过的梯度应用到所有可训练的参数tvars上。
        # 使用get_or_create_global_step()生成全局统一的训练步数。
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        # 定义学习率更新op
        # 以下两句的意思是：如果传入了新的学习速率self._new_lr，那么就把新的学习速率self._new_lr赋给self._lr
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        # 定义操作_lr_update,它使用tf.assign()将_new_lr的值赋给当前的学习速率_lr
        self._lr_update = tf.assign(self._lr, self._new_lr)
        # endregion

    # region 学习速率更新方法
    # 只需要调用 模型m的m.assign_lr(session,新的学习速率值)，assign_lr()内部就把新的学习速率值feed给self._new_lr，执行session._lr_update张量。
    # 相当于调用了上面的这两句，实现了在外部就可以控制模型的学习速率
    def assign_lr(self, session, lr_value):
        # session.run(tf.assign(self._lr, lr_value))
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
    # endregion

    # region 6个属性读取方法
    # 用@property装饰器将返回变量设为只读，防止修改变量引发的问题，方便外部访问。
    # 因为被当做属性使用，访问时不用带括号，如果模型名为m，则访问的形式可以是 m.lr或m.cost或m.initial_state
    @property
    def input(self):
        return self._input
    @property
    def initial_state(self):
        return self._initial_state
    @property
    def cost(self):
        return self._cost
    @property
    def final_state(self):
        return self._final_state
    @property
    def lr(self):
        return self._lr
    @property
    def train_op(self):
        return self._train_op
    # endregion

# endregion

# endregion

# region 2 计算图执行函数
# 训练一个epoch数据的模型执行函数
# 训练时从外部调用run_epoch(session, m, eval_op=m.train_op,verbose=True)，输出epoch的进度 + perplexity + 训练速度 ，并返回训练结果perplexity。
# 验证或测试时从外部调用run_epoch(session, model），只返回验证或测试结果perplexity。
# 本函数体内，执行了两次session，分别是session.run(model.initial_state)和session.run(fetches, feed_dict)
def run_epoch(session, model, eval_op=None, verbose=False):
  start_time = time.time()
  costs = 0.0
  iters = 0
  # 执行model.initial_state获得初始状态，即PTBModel的cell.zero_state(batch_size, tf.float32)值
  state = session.run(model.initial_state)

  # region 以下是拿到cost和state
  fetches = {
      "cost": model.cost,               # 即张量m.cost
      "final_state": model.final_state, # 即张量m.final_state
  }
  # 只有在训练时才能进入这句if语句
  if eval_op is not None:
    fetches["eval_op"] = eval_op        # 传入的是m.train_op,即m内的张量self._train_op
  # 以下的for循环，是为了计算perplexity
  for step in range(model.input.epoch_size):  # 次数为epoch_size
    # 每次循环前，生成训练用的空feed_dict
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h
    # 通过上面这个for循环，将全部的LSTM单元的state加入到feed_dict中
    # 传入feed_dict,执行fetches，对网络进行一次训练
    # fetches之所以能被且必须被执行，是因为fetches字典里每个元素的value都是tensor，
    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]
  # endregion

    costs += cost
    iters += model.input.num_steps
    # 只有训练时（传入的verbose才是True），才能进入下面的if和print语句，即每完成10%的epoch，就进行一次结果的展示。
    # 如果不是训练，则跳过if+print，结束本次循环，进入下一个step循环，直到epoch_size跑完。
    if verbose and step % (model.input.epoch_size // 10) == 10:
      # perplexity即平均cost的自然常数指数，越低代表模型的输出概率分布在预测样本上越好
      print("%.3f perplexity: %.3f speed: %.0f wps" %
             (step * 1.0 / model.input.epoch_size,                       # 当前epoch的进度
             np.exp(costs / iters),                                      # perplexity
             iters * model.input.batch_size / (time.time() - start_time) # 训练速度（每秒的单词数）
             )
           )
  # 返回perplexity，作为本函数run_epoch()结果。
  return np.exp(costs / iters)
# endregion

# tf.Graph()内部调用PTBInput()和PTBModel()，初始化三种数据对象和模型对象，为下文sv的训练测试做好了准备。
with tf.Graph().as_default():

    # region 3 构建计算图
    # 设置参数的初始化器，参数范围在-init_scale,init_scale之间。
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                              config.init_scale)
    # region 3.1 构建训练模型m
    with tf.name_scope("Train"):
        train_input = PTBInput(config=config, data=train_data, name="TrainInput")
    with tf.variable_scope("Model", reuse=None, initializer=initializer):# 注意reuse
        m = PTBModel(is_training=True, config=config, input_=train_input)
        #tf.scalar_summary("Training Loss", m.cost)
        #tf.scalar_summary("Learning Rate", m.lr)
    # endregion

    # region 3.2 构建验证模型mvalid
    with tf.name_scope("Valid"):
        valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
        # 运用之前训练好的模型的参数来测试他们的效果，故定义reuse = True。
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
        #tf.scalar_summary("Validation Loss", mvalid.cost)
    # endregion

    # region 3.3 构建测试模型mtest
    with tf.name_scope("Test"):
        test_input = PTBInput(config=eval_config, data=test_data, name="TestInput")
    with tf.variable_scope("Model", reuse=True, initializer=initializer):
        # 运用之前训练好的模型的参数来测试他们的效果，故定义reuse=True。
        mtest = PTBModel(is_training=False, config=eval_config,
                       input_=test_input)
    # endregion
    # endregion

    # region 4 执行计算图
    # 创建训练的管理器sv
    # 在sv里，主要负责输出：每轮的学习率，每轮训练后的自然常数指数（run_epoch()返回值），每轮验证时的自然常数指数（run_epoch()返回值）。
    # 在max_max_epoch（例如13轮）结束后，输出测试的自然常数指数。
    # 在训练阶段，run_epoch()会输出“当前epoch的进度   perplexity  训练速度（每秒的单词数）”。
    sv = tf.train.Supervisor()
    with sv.managed_session() as session:# 创建默认的session
        for i in range(config.max_max_epoch):# 前面config = SmallConfig()里max_max_epoch=13，即总共可以训练的epoch数

          # region 执行更新学习速率，并输出显示
          # 先计算累计的学习速率衰减值。
          # max_epoch被初始化为4，则前4轮lr_decay=1,只有当到第5轮（i=4）时，lr_decay=0.5**1=0.5,之后以0.5倍加速衰减。
          lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
          # 更新学习速率。用初始学习速率×累计的衰减
          # m是tf.Graph()中定义的训练模型，assigh_lr()是PTBModel()函数中的方法。
          m.assign_lr(session, config.learning_rate * lr_decay)
          # 执行一个epoch的训练，并输出当前的学习速率。
          # 例如第2轮，即输出 Epoch: 2 Learning rate: 1.000
          print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr))) # 即通过PTBModel()中@property修饰的lr()来获取
          # endregion

          # region 4.1 执行训练
          # 输出训练集上的perplexity，即平均cost的自然常数指数
          train_perplexity = run_epoch(session, m, eval_op=m.train_op,
                                       verbose=True)
          print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
          # endregion

          # region 4.2 执行验证
          # 输出验证集上的perplexity
          valid_perplexity = run_epoch(session, mvalid)
          print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
          # endregion

    # region 4.3 执行测试
    #完成所有循环后，计算并输出测试集上的perplexity
    test_perplexity = run_epoch(session, mtest)
    print("Test Perplexity: %.3f" % test_perplexity)
    # endregion

    # endregion
