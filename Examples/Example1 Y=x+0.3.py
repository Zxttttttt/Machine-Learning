import tensorflow as tf
import numpy as np

#create data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

###create tensorflow structure start###
Weights=tf.Variable(tf.random_uniform([1],-0.1,1.0))
biases=tf.Variable(tf.zeros([1]))

y=Weights*x_data+biases

loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()
###create tensorflow structure end###

sess=tf.Session()
sess.run(init) ##Very important

for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biases))

##result：
# 0 [0.8230214] [-0.15717766]
# 20 [0.30567262] [0.18546672]
# 40 [0.15668829] [0.2684319]
# 60 [0.11562467] [0.29129907]
# 80 [0.10430654] [0.29760182]
# 100 [0.101187] [0.299339]
# 120 [0.10032716] [0.29981783]
# 140 [0.10009018] [0.2999498]
# 160 [0.10002486] [0.29998618]
# 180 [0.10000688] [0.2999962]
# 200 [0.1000019] [0.29999897]
