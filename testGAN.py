import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

discriminator_batch = 1

X = tf.placeholder(tf.float32, shape=[None, discriminator_batch*784])

D_W1 = tf.Variable(xavier_init([discriminator_batch*784, 2*128]))
D_b1 = tf.Variable(tf.zeros(shape=[2*128]))

D_W2 = tf.Variable(xavier_init([2*128, 128]))
D_b2 = tf.Variable(tf.zeros(shape=[128]))

D_W3 = tf.Variable(xavier_init([128, 1]))
D_b3 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


Z = tf.placeholder(tf.float32, shape=[None, discriminator_batch*100])

G_W1 = tf.Variable(xavier_init([100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128,784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z,G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def condense_data(data, num_points=100):
    data = np.array(data)
    n = len(data)
    if n <= num_points:
        return np.arange(1, n+1), data, np.zeros_like(data)
    n_div = (n//num_points)*num_points
    divdata = data[0:n_div]
    rest = data[n_div:]
    x = np.arange(1, n_div+1)
    x = x.reshape(num_points, n//num_points)
    x = np.average(x, axis=1)
    np.append(x, np.mean(np.arange(n_div, n+1)))
    divdata = divdata.reshape(num_points, n//num_points)
    avg = np.average(divdata, axis=1)
    std = np.std(divdata, axis=1)
    np.append(avg, np.mean(rest))
    np.append(std, np.std(rest))
    return x, avg, std

Zlist = tf.split(Z, num_or_size_splits=discriminator_batch, axis=1)
G_sample = tf.concat([generator(Zi) for Zi in Zlist],1)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mb_size = 128
Z_dim = 100
export_dir = 'out/%s/'%time.strftime('%Y%m%d-%H%M%S')
if os.path.exists(export_dir):
    print('Export path %s already exists!' % export_dir)
    quit()
export_model = export_dir+'model/'
export_result = export_dir+'result/'
export_vars = export_dir+'variables/'

for path in [export_dir, export_result, export_vars]:
    if not os.path.exists(path):
        os.makedirs(path)

builder = tf.saved_model.builder.SavedModelBuilder(export_model)
saver = tf.train.Saver(max_to_keep=None)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.TRAINING])
builder.save(as_text=True)

i = 0
G_losses= []
D_losses = []

for it in range(10**6):
    if it % 1e3 == 0:
        samples = sess.run(generator(tf.slice(Z, [0, 0], [16, 100])), feed_dict={Z: sample_Z(16, discriminator_batch*Z_dim)})

        fig = plot(samples)
        plt.savefig(export_result+'{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        plt.close(fig)

        save_path = saver.save(sess, export_vars+'{}.ckpt'.format(str(i).zfill(3)))
        print('Model saved in file: %s' % save_path)

        i += 1

    X_mb = np.concatenate([mnist.train.next_batch(mb_size)[0] for _ in range(discriminator_batch)], 1)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, discriminator_batch*Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, discriminator_batch*Z_dim)})

    D_losses.append(D_loss_curr)
    G_losses.append(G_loss_curr)

    if it % 1e3 == 0:
        x, avg, std = condense_data(D_losses)
        plt.semilogy(np.arange(1, len(D_losses)+1), D_losses, label='discriminator loss', color='b')
        plt.semilogy(x, avg, color='r')
        plt.semilogy(x, avg+std, color='r')
        plt.semilogy(x, avg-std, color='r')
        x, avg, std = condense_data(G_losses)
        plt.semilogy(np.arange(1, len(G_losses)+1), G_losses, label='generator loss', color='g')
        plt.semilogy(x, avg, color='m')
        plt.semilogy(x, avg+std, color='m')
        plt.semilogy(x, avg-std, color='m')
        plt.semilogy(np.arange(1, len(G_losses)+1), np.ones_like(D_losses)*np.log(2))
        plt.legend()
        plt.ylim((1e-2,1e1))
        plt.savefig(export_dir+'evolution.png')
        plt.close()
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'.format(D_loss_curr))
        print('G loss: {:.4}'.format(G_loss_curr))
        print()
