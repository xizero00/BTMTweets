import numpy as np
from itertools import combinations, chain
from tqdm import trange


class oBTM:
    """ Biterm Topic Model

        Code and naming is based on this paper http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf
        Thanks to jcapde for providing the code on https://github.com/jcapde/Biterm
    """

    def __init__(self, num_topics, V, alpha=1., beta=0.01, l=0.5):
        self.K = num_topics # topic数
        self.V = V # 所有word组成的数组
        self.alpha = np.full(self.K, alpha)
        self.beta = np.full((len(self.V), self.K), beta)
        self.l = l


    def _gibbs(self, iterations):

        Z = np.zeros(len(self.B), dtype=np.int8) # 每个biterm的topic
        n_wz = np.zeros((len(self.V), self.K), dtype=int) # 每个word在每个topic下的计数
        n_z = np.zeros(self.K, dtype=int) # 每个topic下biterm的计数

        for i, b_i in enumerate(self.B):
            # 初始化biterm的topic
            topic = np.random.choice(self.K, 1)[0] # [0, self.K)内输出1个数字并组成一维数组（ndarray）
            n_wz[b_i[0], topic] += 1
            n_wz[b_i[1], topic] += 1
            n_z[topic] += 1
            Z[i] = topic

        for _ in trange(iterations):
            for i, b_i in enumerate(self.B):
                n_wz[b_i[0], Z[i]] -= 1
                n_wz[b_i[1], Z[i]] -= 1
                n_z[Z[i]] -= 1
                # 固定其他变量更新z变量，z的值实际上是从P(z|其余变量)的分布中采样的值
                # 下面是在计算P(z|其余变量)的分布
                P_w0z = (n_wz[b_i[0], :] + self.beta[b_i[0], :]) / (2 * n_z + self.beta.sum(axis=0))
                P_w1z = (n_wz[b_i[1], :] + self.beta[b_i[1], :]) / (2 * n_z + 1 + self.beta.sum(axis=0))
                P_z = (n_z + self.alpha) * P_w0z * P_w1z
                # P_z = (n_z + self.alpha) * ((n_wz[b_i[0], :] + self.beta[b_i[0], :]) * (n_wz[b_i[1], :] + self.beta[b_i[1], :]) /
                #                            (((n_wz + self.beta).sum(axis=0) + 1) * (n_wz + self.beta).sum(axis=0)))  # todo check out
                P_z = P_z / P_z.sum()
                # 更新新的topic、n(w|z)和n(z)
                Z[i] = np.random.choice(self.K, 1, p=P_z) # p=P_z指定概率进行采样
                n_wz[b_i[0], Z[i]] += 1
                n_wz[b_i[1], Z[i]] += 1
                n_z[Z[i]] += 1


        return n_z, n_wz

    def fit_transform(self, B_d, iterations):
       self.fit(B_d, iterations)
       return self.transform(B_d)

    def fit(self, B_d, iterations):
        self.B = list(chain(*B_d))
        n_z, self.nwz = self._gibbs(iterations)
        # 计算phi(w|z)和theta(z)
        #! 和论文中计算的公式不同
        self.phi_wz = (self.nwz + self.beta) / np.array([(self.nwz + self.beta).sum(axis=0)] * len(self.V))
        self.theta_z = (n_z + self.alpha) / (n_z + self.alpha).sum()
        
        self.alpha += self.l * n_z
        self.beta += self.l * self.nwz


    def transform(self, B_d):
        # B_d的长度为documents数，其中每个documents中有个一维数组存放生成的biterms
        P_zd = np.zeros([len(B_d), self.K]) # shape(num_docs, num_topics)
        for i, d in enumerate(B_d): # 遍历每个doc的biterms，其中d为biterms组成的数组
            
            P_zb = np.zeros([len(d), self.K]) # shape(num_bitems in doc[i], num_topics)
            for j, b in enumerate(d): # 取出其中的一个biterm
                P_zbi = self.theta_z * self.phi_wz[b[0], :] * self.phi_wz[b[1], :]
                P_zb[j] = P_zbi / P_zbi.sum()
            
            # 当doc中只有一个词时无法组成biterm
            if len(d) != 0:
                P_zd[i] = P_zb.sum(axis=0) / P_zb.sum(axis=0).sum()
            else:
                P_zd[i] = np.full((self.K, ), 1.0 / self.K)

        return P_zd


class sBTM(oBTM):

    def __init__(self, S, num_topics, V, alpha=1., beta=0.01, l=0.5):
        oBTM.__init__(self, num_topics, V, alpha, beta, l)
        self.S = S

    def transform(self, B_d):
        # P_zd = super().transform(B_d)

        s_z = np.zeros((len(B_d), self.K, self.S.shape[1]))
        for i, d in enumerate(B_d):
            w_d = list(set(chain(*d)))
            s_z[i] = (self.nwz[w_d][..., None] * self.S[w_d][:, None]).sum(axis=0)

        return s_z
