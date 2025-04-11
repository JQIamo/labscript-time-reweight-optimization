from lyse import *
import matplotlib.pyplot as plt
import numpy as np
import h5py
import zmq
import json

port = 51212
time_seq_labels = [("EVAP_TIME_SEQ", 1),
                   ("EVAP_TIME_SEQ", 2),
                   ("EVAP_TIME_SEQ", 3),
                   ("EVAP_TIME_SEQ", 4),
                   ("EVAP_TIME_SEQ", 5),
                   ("EVAP_TIME_SEQ", 6),
                   ]

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect(f"tcp://127.0.0.1:{port}")

run = Run(path)

globals = run.get_globals()

cost = run.get_result('img_bi_modal_fit', 'bec_ratio')

time_seq = []

for n, i in time_seq_labels:
    time_seq.append(globals[n][i])

res_dict = {
        "time_seq": time_seq,
        "cost": float(cost),
        "bad": int(cost < 0)
        }

print(res_dict)


res_dict_json = json.dumps(res_dict)

socket.send(res_dict_json.encode("utf-8"))
