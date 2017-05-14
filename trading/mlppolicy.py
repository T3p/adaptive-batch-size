import tensorflow as tf
import numpy as np

# Helper to create multilayer perceptron
def create_mlp(input, layer_size, activ=tf.nn.relu):
    if isinstance(activ, list):
        assert len(layer_size) - 1 == len(activ)
    else:
        activ = [activ]*(len(layer_size) - 1)

    x = input
    for i, layer in enumerate(layer_size[:-1]):
        x = tf.layers.dense(x, layer, activation=activ[i])
    x = tf.layers.dense(x, layer_size[-1], activation=None)

    return x

#Helper to get all network weights in a flat list
def flatten_params(weights):
    result = []
    for arr in weights:
        for x in np.nditer(arr):
            result.append(np.asscalar(x))
    return result

class CategoricalPolicy(object):
    """
    Categorical Policy. Given a observation, returns probabilities
    over all possible actions.
    Args:
	space (number of discrete actions)
        policy_layer_size (list of int): layers 
    """

    def __init__(self, n_actions, layers):
        self._n_actions = int(n_actions)
        self._layers = layers

    def _build(self):
        self.tf_scope = 'categorical_policy'
        with tf.variable_scope(self.tf_scope):
            # action_one_hot = tf.one_hot(self._action, int(self._n_actions))
            output_layer = create_mlp(self._state,
                                      self._layers + [self._n_actions],
                                      activ=tf.nn.relu)

            self.action_probs = tf.nn.softmax(output_layer)

            picked_action_prob = tf.reduce_sum(self.action_probs * self._action,
                                               axis=1)

            # Loss
            self.log_prob = tf.log(picked_action_prob + 1e-10)

        return self

    def __call__(self, state_var, action_var):
        self._state = state_var
        self._action = action_var
        return self._build()

    def predict_prob(self, state, action=None, sess=None):
        sess = sess or tf.get_default_session()
        action_probs = sess.run(self.action_probs, {self._state: state})
        if action is not None:
            return action_probs[int(action)]
        else:
            return action_probs

    def predict_action(self, state, sess=None):
        action_probs = self.predict_prob(state, sess=sess)
        assert len(action_probs) == self._n_actions, '{} != {}'.format(len(action_probs), self._n_actions)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action



class NormalPolicy(object):
    """
    Normal Policy. Given a observation, draws an action from a parametrized 
    Normal distribution.
    Args:
        action_size: dimension of the action (number of components)
        mean_layer_size (list of int): mean layers
        std_layer_size (list of int): std layers
        low_action (np.float): lower bound of the action space (default: None)
         For example in gym it is given by env.action_space.low
        high_action (np.float): upper bound of the action space
         For example in gym it is given by env.action_space.high
        min_std (np.array or float): minimal std
        fixed_std: if true, std is constant and equal to min_std
    """

    def __init__(self, action_dim,
                 mean_layer_size, std_layer_size,
                 low_action=None, high_action=None, min_std=1e-5, fixed_std=False):
        self._action_dim = action_dim
        self._mean_layers = mean_layer_size
        self._std_layers = std_layer_size
        self._low_action = low_action
        self._high_action = high_action
        self._min_std = min_std
        self.fixed_std = fixed_std


    def _build(self):
        self.tf_scope = 'normal_policy'
        with tf.variable_scope(self.tf_scope):
            self.mu = create_mlp(self._state, layer_size=self._mean_layers + [self._action_dim])
            # self.mu = tf.reshape(self.mu, shape=[-1, action_size])
            if self.fixed_std:
                self.sigma = np.ones((self._action_dim,),dtype=np.float32)*self._min_std
            else:
                self.sigma = create_mlp(self._state, layer_size= self._std_layers + [self._action_dim])
                # self.sigma = tf.squeeze(self.sigma)
                self.sigma = tf.nn.softplus(self.sigma) + self._min_std
 
            #weight manipulation 
            self.weights = weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,self.tf_scope)
            
            nb_weights = 0
            for w in weights: 
                dim = np.prod(w.shape.as_list())
                nb_weights+=dim
            self.delta_weights = delta_weights = tf.placeholder(tf.float32,[nb_weights])
            self.custom_weights = custom_weights = tf.placeholder(tf.float32,[nb_weights])            

            self.update_weights = []
            base = 0
            for i in range(len(weights)):
                dim = np.prod(weights[i].shape.as_list())
                reshaped_delta = tf.reshape(self.delta_weights[base:base+dim],weights[i].shape)
                self.update_weights.append(weights[i].assign_add(reshaped_delta))
                base+=dim        

            self.reset_weights = []
            base = 0
            for i in range(len(weights)):
                dim = np.prod(weights[i].shape.as_list())
                reshaped_custom = tf.reshape(self.custom_weights[base:base+dim],weights[i].shape)
                self.reset_weights.append(weights[i].assign(reshaped_custom))
                base+=dim        

            self.normal_dist = tf.contrib.distributions.MultivariateNormalDiag(
                self.mu, self.sigma)
            self.picked_action = self.normal_dist._sample_n(1)[0] 

            if self._high_action is not None and self._low_action is not None:
                self.picked_action = tf.clip_by_value(
                    self.picked_action,
                    self._low_action, self._high_action,
                    name='clipped_proposed_action')
                self._action = tf.clip_by_value(
                    self._action,
                    self._low_action, self._high_action,
                    name='clipped_sample_action')

            # Loss
            self.log_prob = self.normal_dist.log_prob(self._action)
            
            #Score (gradient of policy logarithm)
            self.scores = tf.gradients(self.log_prob,self.weights)
            self.flat_scores = []
            for x in self.scores:
                self.flat_scores.append(tf.reshape(x,[-1]))
            self.flat_scores = tf.concat(self.flat_scores,0)

        return self

    def __call__(self, state_var, action_var):
        self._state = state_var
        self._action = action_var
        self.state_size = self._state.shape[-1].value
        return self._build()

    def predict_prob(self, state, action, sess=None):
        sess = sess or tf.get_default_session()
        log_pdf = sess.run(self.log_prob,
                           {self._state: state, self._action: action})
        return np.exp(log_pdf)

    def predict_action(self, state, sess=None):
        reshaped_state = state.reshape([-1, self.state_size])
        sess = sess or tf.get_default_session()
        return sess.run(self.picked_action, {self._state: reshaped_state})
     
    #New functions
    def get_mu(self,state,sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.mu,{self._state: state})

    def get_sigma(self,state,sess=None):
        sess = sess or tf.get_default_session()
        if self.fixed_std:
            return self.sigma
        else:
            return sess.run(self.sigma,{self._state: state})
        
    def log_gradients(self,state,action,sess=None):
        sess = sess or tf.get_default_session()
        gradients = sess.run(self.flat_scores,{self._state: state, self._action: action})
        return gradients

    def get_weights(self,sess=None):
        sess = sess or tf.get_default_session()
        weights = sess.run(self.weights)
        return flatten_params(weights)

    def update(self,delta_w,sess=None):
        sess = sess or tf.get_default_session()
        sess.run(self.update_weights,{self.delta_weights: delta_w})
 
    def reset(self,custom_w,sess=None):
        sess = sess or tf.get_default_session()
        sess.run(self.reset_weights,{self.custom_weights: custom_w})       

    def save_weights(self,idx=0,sess=None):
        weights = self.get_weights(sess)
        np.save('weights/nn_weights'+str(idx),weights)

    def load_weights(self,idx=0,sess=None):
        saved_w = np.load('weights/nn_weights'+str(idx)+'.npy')
        self.reset(saved_w,sess)

if __name__ == "__main__":
    N = 1
    state_size = 2
    action_size = 1
    state = tf.placeholder(tf.float32, [N, state_size])
    action = tf.placeholder(tf.float32, [N, action_size])
    pol = NormalPolicy(action_size, [],
                       [],-1,1,min_std=0.1,
                        fixed_std=True)(state,action)  

    #Compute score and update network weights
    s = np.array([1,1]).reshape((N,state_size))
    a = np.array([0.5]).reshape((N,action_size))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #print pol.get_mu(s), pol.get_sigma(s), '\n'
        w = np.ones(len(pol.get_weights()))
        print pol.get_weights(), '\n'
        #scores = pol.log_gradients(s,a)
        #print scores, '\n'
        #Q = 10
        #alpha = 0.01
        #grads = map(lambda x: x*Q*alpha,scores)
        pol.reset(2.*w)
        print pol.get_weights(), '\n'
        pol.save_weights()
        pol.reset(11.*w)
        print pol.get_weights(), '\n'
        pol.load_weights()
        print pol.get_weights(), '\n'
        


# Example of MultivariateNormalDiag
# import numpy as np
# import tensorflow as tf
#
# # dimensions
# batch_size = 2
# dim = 2
#
# mu = tf.reshape(tf.constant(
#     np.arange(batch_size * dim * 1.0)),
#                 [batch_size, dim])
#
# sigma = tf.reshape(tf.constant(
#     np.arange(batch_size * dim * 1.0) + 1),
#                    [batch_size, dim])
#
# mvn = tf.contrib.distributions.MultivariateNormalDiag(mu, sigma)
#
# with tf.Session() as sess:
#     print(sess.run(mu))
#     print(sess.run(sigma))
#     print(mvn.mean().eval())
#     print(mvn.stddev().eval())
#     x = mvn.mean().eval()
#     print(mvn.prob(x).eval())
