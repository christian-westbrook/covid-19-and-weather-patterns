import os
import argparse
import tensorflow as tf
import json

"""
HOW TO USE:
1) Create Python 3.8 venv:
/usr/bin/python3.8 -m venv ~/py38_venv

2) Activate and install Tensorflow 2.7:
cd py38_venv/
source bin/activate
pip3 install tensorflow

3) ssh to salem, topeka, pierre, providence
activate venv in each shell

4) Run commands in corresponding shells:

(py38_venv) salem:~/cs555$ python tf_distribute_demo.py --type worker --index 0

(py38_venv) topeka:~/cs555$ python tf_distribute_demo.py --type worker --index 1

(py38_venv) pierre:~/cs555$ python tf_distribute_demo.py --type ps --index 0

(py38_venv) providence:~/cs555$ python tf_distribute_demo.py --type chief --index=0

Running --type chief will train the model and evaluate. 
Individual servers can be stopped using Ctrl + \ (it takes a second to dump memory)

"""

parser = argparse.ArgumentParser()
parser.add_argument('--type', choices=['chief', 'ps', 'worker'], required=True)
parser.add_argument('--index', type=int, required=True)

args = parser.parse_args()
chief = ["providence:11888"]
worker = ["salem:11889", "topeka:11891"]
ps = ["pierre:11890"]

# Dump the cluster information to `'TF_CONFIG'`.
tf_config = {
    'cluster': {
        'chief': chief,
        'worker': worker,
        'ps':  ps,
    },
    'task': {'type': 'chief', 'index': 0}
}

os.environ['TF_CONFIG'] = json.dumps(tf_config)
tf.print("calling cluster_resolver")
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()


if args.type == 'worker':
    server = tf.distribute.Server(cluster_resolver.cluster_spec(), job_name="worker", task_index=args.index)
    server.join()
elif args.type == 'ps':
    server = tf.distribute.Server(cluster_resolver.cluster_spec(), job_name="ps", task_index=args.index)
    server.join()
else:

    tf.print("Calling ParameterServerStrategy")
    strategy = tf.distribute.experimental.ParameterServerStrategy(cluster_resolver)

    tf.print("ParameterServerStrategy Done")
    features = [[1., 1.5], [2., 2.5], [3., 3.5]]
    labels = [[0.3], [0.5], [0.7]]
    eval_features = [[4., 4.5], [5., 5.5], [6., 6.5]]
    eval_labels = [[0.8], [0.9], [1.]]

    dataset = tf.data.Dataset.from_tensor_slices(
        (features, labels)).shuffle(10).repeat().batch(64)

    eval_dataset = tf.data.Dataset.from_tensor_slices(
        (eval_features, eval_labels)).repeat().batch(1)

    with strategy.scope():
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.05)
        model.compile(optimizer, "mse")

    tf.print("calling model.fit()")
    model.fit(dataset, epochs=5, steps_per_epoch=10)

    tf.print("calling model.evaluate()")
    model.evaluate(eval_dataset, steps=10, return_dict=True)