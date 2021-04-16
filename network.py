# coding: utf-8

import os
import re
import json
import numpy as np
from draw_spike import DrawSpike


class SpikingNetwork:
    TAU_MP = 20  # / ms
    T_REF = 1  # / ms
    ALPHA = 2
    ETA_W = 0.002
    ETA_TH = ETA_W * 0.1
    SIGMA = 0.5
    #BETA = 0.0001
    #LAMBDA = 0.00000001

    def __init__(self, draw_spike=False):
        self.weights = []
        self.thresholds = []
        self.num_neurons = []
        self.v_mps = []
        self.kappas = []

        self.draw_spike = None
        if draw_spike:
            self.draw_spike = DrawSpike(network=self)

    # add a layer to the network
    def add(self, n, kappa=None):
        self.num_neurons.append(n) # the last element is the number of neurons in the PRESENT layer
        if len(self.num_neurons) == 1: # if the network has just only an input layer
            if kappa is not None:
                print('This kappa is ignored.')
            return
        # the network has at least 1 hidden layer
        self.v_mps.append(np.zeros((n, 1)))

        if kappa is None:
            kappa = 0 # no lateral inhibitory connections
        self.kappas.append(kappa) # lateral inhibitory connections
        
        prev_n = self.num_neurons[len(self.num_neurons) - 2] # the number of neurons in the PREVIOUS layer
        root_3_per_m = np.sqrt(3 / prev_n) # The number of neurons in the previous layer is the number of 
        # synapses of a neuron in the present layer

        # weight and threshold initialization
        self.weights.append(np.random.uniform(-root_3_per_m, root_3_per_m, (n, prev_n)))
        self.thresholds.append(np.ones((n, 1)) * SpikingNetwork.ALPHA * root_3_per_m)

    # time / ms
    def forward(self, x, exposed_time):
        self.spikes = [[] for _ in self.num_neurons]
        self.x_ks = [np.zeros((num_neuron, 1)) for num_neuron in self.num_neurons[:-1]]
        self.a_is = [np.zeros((num_neuron, 1)) for num_neuron in self.num_neurons[1:]]

        import time
        start_time = time.time()
        for t in range(exposed_time):
            input_spike = x # shape x: (784, #test examples)
            for i, (v_mp, spike, weight, threshold, x_k, a_i, kappa) in enumerate(zip(
                    self.v_mps,
                    self.spikes,
                    self.weights,
                    self.thresholds,
                    self.x_ks,
                    self.a_is,
                    self.kappas
            )):
                #if not np.any(input_spike):
                #    break
                spike.append(input_spike)

                x_k = SpikingNetwork._calc_x_k(spike, t) # Need to check t_p < t
                if i > 0:
                    self.a_is[i - 1] = x_k

                tmp_x_k = x_k.copy()
                tmp_x_k[~input_spike] = 0
                # threshold, a_i are vectors. Others are constants
                # elment-wise multiplication
                lateral_inhibition = SpikingNetwork.SIGMA * threshold * kappa * a_i
                lateral_inhibition = lateral_inhibition.sum(axis=0) * np.ones(a_i.shape) - lateral_inhibition
                v_mp = weight @ tmp_x_k - threshold * a_i + lateral_inhibition

                input_spike = np.zeros(v_mp.shape, dtype=bool)
                input_spike[v_mp > threshold] = True
                v_mp -= (v_mp > threshold) * threshold

                self.v_mps[i] = v_mp
                self.spikes[i] = spike
                self.x_ks[i] = x_k

            self.spikes[-1].append(input_spike)
            self.a_is[-1] = SpikingNetwork._calc_x_k(self.spikes[-1], t)

            if self.draw_spike is not None:
                self.draw_spike.update()

        self.t = t

        for i, _ in enumerate(self.v_mps):
            self.v_mps[i] = np.zeros(self.v_mps[i].shape)

        print('forward time: {}'.format(time.time() - start_time))

    def backward(self, y):
        import time
        start_time = time.time()
        self.x_ks = [SpikingNetwork._calc_x_k(spike, self.t) for spike in self.spikes[:-1]]
        self.x_ks = [x_k if x_k.sum() > 0.00001 else np.ones((num_neuron, y.shape[1])) * 10
                     for x_k, num_neuron in zip(self.x_ks, self.num_neurons[:-1])]
        self.a_is = [SpikingNetwork._calc_x_k(spike, self.t) for spike in self.spikes[1:]]
        self.a_is = [a_i if a_i.sum() > 0.00001 else np.ones((num_neuron, y.shape[1])) * 10
                     for a_i, num_neuron in zip(self.a_is, self.num_neurons[1:])]
        print('x_ks and a_is time: {}'.format(time.time() - start_time))

        sharp_spikes = self._calculate_sharp_spikes()
        has_spike_in_output = sharp_spikes.max(axis=0) > 0.0000001
        o_is = np.zeros(sharp_spikes.shape)
        o_is[:, has_spike_in_output] = sharp_spikes[:, has_spike_in_output] / sharp_spikes[:, has_spike_in_output].max(axis=0)
        delta = o_is - y

        m_ls = [np.array(sum(spike), bool).sum(axis=0).astype(float) for spike in self.spikes[:-1]]
        for m_l in m_ls:
            m_l[m_l < 0.00001] = 0.0001

        for i, weight, threshold, x_k, a_i, m_l in zip(
                reversed(range(len(m_ls))),
                reversed(self.weights),
                reversed(self.thresholds),
                reversed(self.x_ks), reversed(self.a_is),
                reversed(m_ls)):
            N_l = weight.shape[0]
            M_l = weight.shape[1]

            '''
            for j, this_has_spike_in_output in enumerate(has_spike_in_output):
                if not this_has_spike_in_output and np.sum(delta[:, j][:, np.newaxis] @ x_k.T[j][np.newaxis, :]) > 0:
                    delta[:, j] *= -1
            '''
            delta_for_weight = -SpikingNetwork.ETA_W * np.sqrt(N_l / m_l) * delta
            if np.all(has_spike_in_output):
                delta_weight = delta_for_weight @ x_k.T
            else:
                import time
                s = time.time()
                delta_weight = sum(map(lambda x: np.absolute(x[0]) if not x[1] else x[0], [
                    (delta_for_weight[:, j][:, np.newaxis] @ x_k.T[j][np.newaxis, :], this_has_spike_in_output)
                    for j, this_has_spike_in_output in enumerate(has_spike_in_output)
                ]))
                print('delta_weight time: {}'.format(time.time() - s))
            delta_threshold = -SpikingNetwork.ETA_TH * np.sqrt(N_l / (m_l * M_l)) * delta * a_i

            if i - 1 >= 0:
                delta = (1 / self.thresholds[i - 1]) / np.sqrt((1 / m_ls[i - 1]) * np.sum((1 / self.thresholds[i - 1]) ** 2)) * np.sqrt(
                    M_l / m_l) * (weight.T @ delta)

            weight += delta_weight - 0.0001 * weight
            #weight /= np.absolute(weight).sum(axis=1)[:, np.newaxis]
            '''
            weight_regularization = np.exp(SpikingNetwork.BETA * (np.sum(weight ** 2, axis=1) - 1))[:, np.newaxis]
            weight_regularization = SpikingNetwork.BETA * SpikingNetwork.LAMBDA * weight * np.concatenate([
            #weight_regularization = 0.5 * SpikingNetwork.LAMBDA * np.concatenate([
                weight_regularization for _ in range(weight.shape[1])], axis=1)
            #print('weight_regularization: {}'.format(weight_regularization))
            #weight -= weight_regularization
            weight += delta_weight - weight_regularization
            '''
            #threshold += delta_threshold.mean(axis=1)[:, np.newaxis]

            '''
            weight_regularization = np.exp(SpikingNetwork.BETA * (np.sum(weight ** 2, axis=1) - 1))[:, np.newaxis]
            weight_regularization = SpikingNetwork.BETA * SpikingNetwork.LAMBDA * weight * np.concatenate([
            #weight_regularization = 0.5 * SpikingNetwork.LAMBDA * np.concatenate([
                weight_regularization for _ in range(weight.shape[1])], axis=1)
            #print('weight_regularization: {}'.format(weight_regularization))
            weight -= weight_regularization
            '''
            #weight += delta_weight - weight_regularization
            #threshold += delta_threshold

    def infer(self, display_no_spike=False):
        sharp_spikes = self._calculate_sharp_spikes() * 100
        no_spikes = sharp_spikes.max(axis=0) < 0.000001
        if len(no_spikes) > 0 and display_no_spike:
            print('No spike... in {}'.format(no_spikes))
        max_sharp_spike = np.max(sharp_spikes, axis=0)
        return np.exp(sharp_spikes - max_sharp_spike) / np.sum(np.exp(sharp_spikes - max_sharp_spike), axis=0)

    def save(self, path=None):
        if path is None:
            path = os.path.join('./models', '{}.npz'.format(SpikingNetwork._get_latest_model_number() + 1))

        content = {
            'w{}'.format(i): weight
            for i, weight in enumerate(self.weights)
        }
        content.update({
            't{}'.format(i): threshold
            for i, threshold in enumerate(self.thresholds)
        })
        np.savez(path, **content)

    def load(self, path=None):
        if path is None:
            latest_model_number = SpikingNetwork._get_latest_model_number()
            if latest_model_number == 0:
                raise Exception('There is no numbered model!')
            path = os.path.join('./models', '{}.npz'.format(latest_model_number))

        content = np.load(path)
        self.weights = [content['w{}'.format(i)] for i, _ in enumerate(self.weights)]
        self.thresholds = [content['t{}'.format(i)] for i, _ in enumerate(self.thresholds)]

    def _calculate_sharp_spikes(self):
        return sum(self.spikes[-1])

    @classmethod
    def _calc_x_k(cls, spike, t):
        # If fired, then compute the sum. Otherwise, do nothing
        # Imagine a 'spike' is like a list [False,True,True,False,...], True is an active neuron, (784, 1)
        # Only considering active neurons
        return sum([np.exp((t_p - t) / cls.TAU_MP) * fire for t_p, fire in enumerate(spike)])

    @staticmethod
    def _get_latest_model_number():
        path = './models'
        if not os.path.exists(path):
            os.mkdir(path)
        models = list(filter(lambda y: y is not None, map(lambda x: re.match('[0-9]+.npz', x), os.listdir(path))))
        if len(models) == 0:
            return 0
        else:
            return max(map(lambda x: int(x.group().split('.')[0]), models))
