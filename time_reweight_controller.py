import numpy as np
import json
import zmq
import zmq.asyncio
import logging
import asyncio
import threading
import pandas as pd
import itertools
from scipy.stats import linregress
from asyncio import Queue, Event

import matplotlib.pyplot as plt
import runmanager.remote as rm

import matplotlib
matplotlib.use('Qt5Agg')

bind_port = 51212

ctx = zmq.asyncio.Context()

time_seq_labels = [("EVAP_TIME_SEQ", 1),
                   ("EVAP_TIME_SEQ", 2),
                   ("EVAP_TIME_SEQ", 3),
                   ("EVAP_TIME_SEQ", 4),
                   ("EVAP_TIME_SEQ", 5),
                   ("EVAP_TIME_SEQ", 6)]

init_globals_dict = {
    "EVAP_TIME_SEQ": (0.05 , 1.95 , 2.845, 1.25 , 1.845, 0.975, 2.195)
}

iterations = 4
total_time_redistrib_each_iter = [0.3]

time_scan_range_frac = 0.5  # scan each segment with +/- 50% time range
#time_scan_range = 0.7  # scan each segment with +/- 50% time range
time_scan_points = 4
min_seg_time = 0.1
scan_repeat = 2
shuffle = True

current_time_seq = []

logger = logging.getLogger("main")

logger.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.DEBUG)

continue_flag = False

vis_axes, vis_costs, vis_res = None, None, None
vis_event = None

def run_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def compute_next_step(df, last_time_seq, generation_ind, visualize=False):
    global vis_axes, vis_costs, vis_res, vis_event

    slopes = np.zeros(len(last_time_seq))
    slope_errs = np.zeros(len(last_time_seq))
    fit_res_arr = []

    ind_range = np.arange(len(last_time_seq))

    axes = []
    costs = []

    for ind in range(len(last_time_seq)):
        other_inds = ind_range[(ind_range != ind)]

        sel = np.all([(df[f"t_{ind}"] == last_time_seq[ind]) for ind in other_inds], axis=0)

        df_sel = df[sel]
        df_sel = df_sel[~df_sel["bad"]]

        axis = df_sel[f"t_{ind}"]
        cost = df_sel["cost"]

        axes.append(list(axis))
        costs.append(list(cost))

        res = linregress(axis, cost)
        slopes[ind] = res.slope
        slope_errs[ind] = res.stderr
        fit_res_arr.append(res)

    logger.info("Slope calculated: [ " + ", ".join([ f"{s:.3f} +/- {e:.3f}" for s, e in zip(slopes, slope_errs) ]) + " ]")

    if visualize:
        vis_axes, vis_costs, vis_res = axes, costs, fit_res_arr
        vis_event.set()

    sort_ind = np.argsort(slopes)
    num_donors = len(sort_ind) // 2

    donor_inds = list(sort_ind[:num_donors])
    recipient_inds = list(sort_ind[num_donors:])

    # reject donors and recipients that are ambiguious

    while len(donor_inds) and len(recipient_inds):
        up_bounds = np.array([r.slope + r.stderr for r in fit_res_arr])
        low_bounds = np.array([r.slope - r.stderr for r in fit_res_arr])
        donor_safe_up_bounds = up_bounds[donor_inds]
        recipient_safe_low_bounds = low_bounds[recipient_inds]

        most_ambiguious_donor_ind = donor_inds[np.argmax(donor_safe_up_bounds)]
        most_ambiguious_recipient_ind = recipient_inds[np.argmin(recipient_safe_low_bounds)]

        if (up_bounds[most_ambiguious_donor_ind] > low_bounds[most_ambiguious_recipient_ind]):
            donor_inds.remove(most_ambiguious_donor_ind)
            recipient_inds.remove(most_ambiguious_recipient_ind)
        else:
            break

    if len(donor_inds) == 0:
        logger.info(f"No unambiguious changes can be made. Terminate...")
        return None

    time_to_redistrib = total_time_redistrib_each_iter[generation_ind] * len(donor_inds) / num_donors

    donor_weights = 1/slopes[donor_inds]
    donor_amounts = -1 * donor_weights * time_to_redistrib / np.sum(donor_weights)

    recipient_weights = slopes[recipient_inds]
    recipient_amounts = recipient_weights * time_to_redistrib / np.sum(recipient_weights)

    dt_seq = np.zeros(len(last_time_seq))
    dt_seq[donor_inds] = donor_amounts
    dt_seq[recipient_inds] = recipient_amounts

    logger.info(f"Proposing changes to time sequence: {dt_seq}")

    next_t_seq = last_time_seq + dt_seq

    logger.info(f"Proposing new time sequence: {next_t_seq}")

    return next_t_seq


def visualize_loop():
    global vis_axes, vis_costs, vis_res, vis_event

    while vis_event.wait():
        logger.info(f"Received visualization request")
        assert vis_axes is not None and vis_costs is not None and vis_res is not None \
            and vis_event is not None

        vis_event.clear()
        n_plots = len(vis_axes)

        fig2 = plt.figure(figsize=(5, 4))

        plt.bar(range(len(vis_res)), [res.slope for res in vis_res], yerr=[res.stderr for res in vis_res])
        plt.gca().set_xticklabels([f"{label}[{ind}]" for label, ind in time_seq_labels])
        plt.xticks(rotation=45, ha='left')

        fig, axs = plt.subplots(n_plots, 1, figsize=(5, 1.5*n_plots))

        costs_flatten = list(itertools.chain.from_iterable(vis_costs))
        costs_max = np.max(costs_flatten)
        costs_min = np.min(costs_flatten)

        for i in range(n_plots):
            label, ind = time_seq_labels[i]
            axs[i].scatter(vis_axes[i], vis_costs[i], marker='x', label=f"{label}[{ind}]")
            axs[i].plot(vis_axes[i], np.asarray(vis_axes[i])*vis_res[i].slope + vis_res[i].intercept, label=f"{label}[{ind}]")
            axs[i].set_ylim(costs_min, costs_max)
            axs[i].legend()

        plt.show()


def make_scan_schedule(time_seq):
    time_segs = { k: [] for k in range(len(time_seq)) }

    for i in range(len(time_seq)):
        time_seq_scan = list(time_seq)
        for dt_frac in (1 + np.linspace(-time_scan_range_frac/2, time_scan_range_frac/2, time_scan_points)):
            time_seq_scan = list(time_seq)
            time_seq_scan[i] = max(dt_frac*time_seq_scan[i], min_seg_time)

        #for dt_frac in (np.linspace(-time_scan_range, time_scan_range, time_scan_points)):
        #    time_seq_scan = list(time_seq)
        #    time_seq_scan[i] = max(time_seq_scan[i] + dt_frac, min_seg_time)

            [ time_segs[ind].append(v) for ind, v in enumerate(time_seq_scan) ]


    df_dict = {
            "executed": False,
            "cost": np.nan,
            "bad": True,
            }

    for k, v in time_segs.items():
        df_dict[f"t_{k}"] = v

    df = pd.DataFrame(df_dict)

    if scan_repeat > 1:
        df_repeat = pd.concat([df]*int(scan_repeat), ignore_index=True).reset_index()
        return df_repeat
    else:
        return df


async def emit_shots(df_sche):
    global shuffle

    df_sche_sel = df_sche[df_sche["executed"] == False]

    if shuffle:
        df_sche_sel = df_sche_sel.sample(frac=1)

    for rind, row in df_sche_sel.iterrows():
        name = ""

        glob = init_globals_dict.copy()

        for ind, (name, gind) in enumerate(time_seq_labels):
            time_seg = list(glob[name])
            time_seg[gind] = row[f"t_{ind}"]
            glob[name] = tuple(time_seg)

        logger.info(f"Emit shots with global: {glob}")

        rm.set_globals(glob)
        rm.engage()

        await asyncio.sleep(0.1)


async def recv_and_process(socket, df_sche, current_time_seq):
    global continue_flag

    while not np.all(df_sche["executed"]):
        msg_b = await socket.recv()
        msg = msg_b.decode("utf-8")

        msg_dict = json.loads(msg)

        shot_time_seq = np.array(msg_dict["time_seq"])
        shot_cost = msg_dict["cost"]
        bad = bool(msg_dict["bad"])

        logger.info(f"Get shot with time sequence {shot_time_seq} and cost {shot_cost} (bad = {bad}).")

        sel_ = np.all([np.isclose(df_sche[f"t_{i}"], t_seg) for i, t_seg in enumerate(shot_time_seq)], axis=0)

        if not np.sum(sel_):
            logger.warning("This shot is not in the schedule, ignore.")
            continue

        sel = sel_ & (df_sche["executed"] == False)

        if not np.sum(sel):
            sel = sel_ & (df_sche["bad"] == True)
            if not np.sum(sel):
                logger.warning("This shot has been executed, ignore.")
                continue
            else:
                logger.warning("Replaced previous bad shot")

        ind = df_sche[sel].first_valid_index()

        df_sche.at[ind, "executed"] = True
        df_sche.at[ind, "cost"] = shot_cost
        df_sche.at[ind, "bad"] = bad

        remaining = np.sum(df_sche["executed"] == False)
        logger.info(f"Mark schedule item {ind} as executed. Remaining shots: {remaining}")
        logger.info(df_sche[df_sche["executed"] == False])

    continue_flag = True


async def main():
    global continue_flag

    socket = ctx.socket(zmq.PULL)
    socket.bind(f"tcp://0.0.0.0:{bind_port}")
    event = asyncio.Event()

    current_time_seq = [init_globals_dict[n][i] for n, i in time_seq_labels]

    for ind_iter in range(iterations):
         try:
             df_sche = make_scan_schedule(current_time_seq)
         except e:
             print(e)
             return
         df_sche = make_scan_schedule(current_time_seq)

         emit_task = asyncio.create_task(emit_shots(df_sche))
         recv_task = asyncio.create_task(recv_and_process(socket, df_sche, current_time_seq))
         event_get_task = asyncio.create_task(event.wait())


         done_tasks, pending_tasks = await asyncio.wait(
                 [emit_task, recv_task, event_get_task],
                 return_when=asyncio.FIRST_COMPLETED
                 )

         while pending_tasks:
             if event.is_set():
                 logger.info("Termination signal received, trying to shut down gracefully...")
                 [i.cancel() for i in pending_tasks]
                 await asyncio.gather(*pending_tasks, return_exceptions=True)
                 logger.info("Shutdown completed.")
                 break
             else:
                 for t in done_tasks:
                     if t.exception():
                         logger.exception(t.exception())

                 if continue_flag:
                     continue_flag = False
                     [i.cancel() for i in pending_tasks]
                     break
                 else:
                     done_tasks, pending_tasks = await asyncio.wait(
                             pending_tasks, return_when=asyncio.FIRST_COMPLETED)

         try:
             current_time_seq = compute_next_step(df_sche, current_time_seq, ind_iter, visualize=True)
         except Exception as e:
             logger.exception(e)
             return

         if current_time_seq is None:
            return

    logger.info("Finished.")



if __name__ == "__main__":
    vis_event = threading.Event()

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=run_event_loop, args=(loop,))
    thread.start()

    future = asyncio.run_coroutine_threadsafe(main(), loop)

    visualize_loop()

    future.result()
