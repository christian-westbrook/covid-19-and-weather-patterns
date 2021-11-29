import os
import argparse
import tensorflow as tf
import yaml
import json

"""
HOW TO USE:

1) Create Python 3.8 venv:
/usr/bin/python3.8 -m venv ~/py38_venv

2) Activate and install Tensorflow 2.7:
cd py38_venv/
source bin/activate
pip3 install tensorflow

3) ssh to hosts in ../distribute.yaml, note the index number that they appear in each array.
activate venv in each shell

4) Run commands in corresponding shells:

(py38_venv) salem:~/cs555$ python start_worker.py --type worker --index 0

(py38_venv) topeka:~/cs555$ python start_worker.py --type worker --index 1

(py38_venv) pierre:~/cs555$ python start_worker.py --type ps --index 0

Individual servers can be stopped using Ctrl + \ (it takes a second to dump memory)
"""

parser = argparse.ArgumentParser()
parser.add_argument('--type', choices=['ps', 'worker'], required=True)
parser.add_argument('--index', type=int, required=True)
args = parser.parse_args()

with open('../distributed.yaml') as stream:
    distributed_conf = yaml.safe_load(stream)
tf_config = {
    'cluster': {
        'chief': distributed_conf['chief'],
        'worker': distributed_conf['worker'],
        'ps': distributed_conf['ps'],
    },
    'task': {'type': 'chief', 'index': 0}
}

os.environ['TF_CONFIG'] = json.dumps(tf_config)
cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
if args.type == 'worker':
    server = tf.distribute.Server(cluster_resolver.cluster_spec(), job_name="worker", task_index=args.index)
    server.join()
elif args.type == 'ps':
    server = tf.distribute.Server(cluster_resolver.cluster_spec(), job_name="ps", task_index=args.index)
    server.join()
