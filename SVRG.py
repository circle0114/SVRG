import numpy as np
import mxnet as mx
from mxnet.test_utils import same
from mxnet.contrib.svrg_optimization.svrg_module import SVRGModule
from mxnet.contrib.svrg_optimization.svrg_optimizer import _SVRGOptimizer


def create_network():

    train_data = np.random.randint(1, 5, [1000, 2])
    weights = np.array([1.0, 2.0])
    train_label = train_data.dot(weights)

    batch_size = 32

    di = mx.io.NDArrayIter(train_data, train_label, batch_size=batch_size, shuffle=True, label_name='lin_reg_label')
    X = mx.sym.Variable('data')
    Y = mx.symbol.Variable('lin_reg_label')
    fully_connected_layer = mx.sym.FullyConnected(data=X, name='fc1', num_hidden=1)
    lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer, label=Y, name="lro")

    mod = SVRGModule(
        symbol=lro,
        data_names=['data'],
        label_names=['lin_reg_label'], update_freq=2
    )

    mod.bind(data_shapes=di.provide_data, label_shapes=di.provide_label)
    mod.init_params(initializer=mx.init.Uniform(0.01), allow_missing=False,
                    force_init=False, allow_extra=False)

    return di, mod


def test_init_svrg_optimizer():
    _, mod = create_network()

    kv = mx.kv.create('local')
    mod.init_optimizer(kvstore=kv, optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
                       force_init=False)

    assert type(mod._optimizer).__name__ == _SVRGOptimizer.__name__


def test_svrg_optimizer_constructor():
    kv = mx.kv.create('local')
    svrg_optimizer = _SVRGOptimizer(default_optimizer='sgd', learning_rate=-1.0)
    kv.set_optimizer(svrg_optimizer)

    assert svrg_optimizer.default_opt.lr == -1.0


def test_kvstore_init_aux_keys():
    param_idx2name = {0: "weight", 1: "weight_full"}

    svrg_optimizer = _SVRGOptimizer(default_optimizer='sgd', param_idx2name= param_idx2name, learning_rate=1.0)
    kv = mx.kv.create('local')
    kv.set_optimizer(svrg_optimizer)

    # Use default sgd optimizer
    param_weight_init = mx.nd.array([0, 0, 0])
    param_weight_update = mx.nd.array([1, 1, 1])

    kv.init(0, param_weight_init)
    kv.push(0, param_weight_update)
    kv.pull(0, param_weight_init)

    param_weight_full_init = mx.nd.array([1, 1, 1])
    param_weight_full_update = mx.nd.array([2, 2, 2])

    # Use AssignmentOptimizer
    kv.init(1, param_weight_full_init)
    kv.push(1, param_weight_full_update)
    kv.pull(1, param_weight_full_init)

    # updated weights using default sgd optimizer
    assert same(param_weight_init.asnumpy(), np.array([-1, -1, -1]))
    # updated with AssignmentOptimizer
    assert same(param_weight_full_init.asnumpy(), np.array([2, 2, 2]))


if __name__ == "__main__":
    import nose
    nose.runmodule()