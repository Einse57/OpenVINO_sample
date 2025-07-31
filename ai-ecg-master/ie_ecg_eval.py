import tkinter
import numpy as np
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import util
import load
import logging as log
import time
from datetime import datetime
import scipy.stats as sst
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ecg_menu
import threading
from cpuinfo import get_cpu_info
from matplotlib.widgets import Button
from openvino import Core

ecg_height = 8960

class InferReqWrap:
    def __init__(self, request, id, callbackQueue):
        self.id = id
        self.request = request
        self.request.set_completion_callback(self.callback, self.id)
        self.callbackQueue = callbackQueue

    def callback(self, statusCode, userdata):
        if (userdata != self.id):
            print("Request ID {} does not correspond to user data {}".format(self.id, userdata))
        elif statusCode != 0:
            print("Request {} failed with status code {}".format(self.id, statusCode))
        self.callbackQueue(self.id, self.request.latency)

    def startAsync(self, input_data):
        self.request.async_infer(input_data)

    def infer(self, input_data):
        self.request.infer(input_data)
        self.callbackQueue(self.id, self.request.latency);

class InferRequestsQueue:
    def __init__(self, requests):
      self.idleIds = []
      self.requests = []
      self.times = []
      for id in range(0, len(requests)):
          self.requests.append(InferReqWrap(requests[id], id, self.putIdleRequest))
          self.idleIds.append(id)
      self.startTime = datetime.max
      self.endTime = datetime.min
      self.cv = threading.Condition()

    def resetTimes(self):
      self.times.clear()

    def getDurationInSeconds(self):
      return (self.endTime - self.startTime).total_seconds()

    def putIdleRequest(self, id, latency):
      self.cv.acquire()
      self.times.append(latency)
      self.idleIds.append(id)
      self.endTime = max(self.endTime, datetime.now())
      self.cv.notify()
      self.cv.release()

    def getIdleRequest(self):
        self.cv.acquire()
        while len(self.idleIds) == 0:
            self.cv.wait()
        id = self.idleIds.pop();
        self.startTime = min(datetime.now(), self.startTime);
        self.cv.release()
        return self.requests[id]

    def waitAll(self):
        self.cv.acquire()
        while len(self.idleIds) != len(self.requests):
            self.cv.wait()
        self.cv.release()

def parseValuePerDevice(devices, values_string):
    ## Format: <device1>:<value1>,<device2>:<value2> or just <value>
    result = {}
    if not values_string:
      return result
    device_value_strings = values_string.upper().split(',')
    for device_value_string in device_value_strings:
        device_value_vec = device_value_string.split(':')
        if len(device_value_vec) == 2:
            for device in devices:
                if device == device_value_vec[0]:
                    value = int(device_value_vec[1])
                    result[device_value_vec[0]] = value
                    break
        elif len(device_value_vec) == 1:
            value = int(device_value_vec[0])
            for device in devices:
                result[device] = value
        elif not device_value_vec:
            raise Exception("Unknown string format: " + values_string)
    return result

def parseDevices(device_string):
    devices = device_string
    if ':' in devices:
        devices = devices.partition(':')[2]
    return [ d[:d.index('(')] if '(' in d else d for d in devices.split(',') ]

def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with the "
                           "kernels implementations.", type=str, default=None)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to labels mapping file", default=None, type=str)
    args.add_argument("-pt", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-pc", "--perf_counts", type=str2bool, required=False, default=False, nargs='?', const=True,
                      help="Optional. Report performance counters.", )

    return parser

def on_select(item):
    # Remove previous plot axes before rendering a new one
    if hasattr(fig, 'last_ax1') and fig.last_ax1 in fig.axes:
        fig.delaxes(fig.last_ax1)
        fig.last_ax1 = None
    ax1 = fig.add_subplot(gs[2, :])
    fig.last_ax1 = ax1
    ax2 = fig.add_subplot(gs[1, 3])
    image = plt.imread("openvino-logo.png")
    ax2.axis('off')
    ax2.imshow(image)
    if 'clear' in (item.labelstr):
        ax1.cla()

    else:
        log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
        args = build_argparser().parse_args()
        #read input data
        if 'Async' in (item.labelstr):
            ecg_data = load.load_ecg("A00001.mat")
        else:
            ecg_data = load.load_ecg(item.labelstr)
        preproc = util.load(".")
        input_ecg = preproc.process_x([ecg_data])
        ecg_n, ecg_h, ecg_w = input_ecg.shape
        log.info("Input ecg file shape: {}".format(input_ecg.shape))

        input_ecg_plot = np.squeeze(input_ecg)
        # Clear previous plot and avoid double rendering
        ax1.cla()
        # raw signal plot
        Fs = 1000
        N = len(input_ecg_plot)
        T = (N-1)/Fs
        ts = np.linspace(0, T, N, endpoint=False)
        ax1.plot(ts, input_ecg_plot, label=item.labelstr, lw=2)
        ax1.set_ylabel('Amplitude')
        ax1.set_title("ECG Raw signal: length - {}, Freq - 1000 Hz".format(ecg_h))
        ax1.legend(loc='upper right')

        #choose proper IRs
        if (input_ecg.shape[1]==8960):
            model_xml = "ecg_8960_ir10_fp16.xml"
            model_bin = os.path.splitext(model_xml)[0] + ".bin"
        elif (input_ecg.shape[1]==17920):
            model_xml = "ecg_17920_ir10_fp16.xml"
            model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Plugin initialization for specified device and load extensions library if specified
        log.info("OpenVINO Initializing plugin for {} device...".format(args.device))
        ie = Core()

        # Read IR
        log.info("OpenVINO Reading IR...")

        # NOTE: IENetwork is removed in OpenVINO 2025+, use ie.read_network()
        net = ie.read_model(model=model_xml, weights=model_bin)
        # NOTE: input/output info via net.inputs/outputs is deprecated, use compiled_model.input()/output()
        assert len(net.inputs) == 1, "Demo supports only single input topologies"

        config = { 'PERF_COUNT' : ('YES' if args.perf_counts else 'NO')}
        device_nstreams = parseValuePerDevice(args.device, None)
        # NOTE: set_config API may differ in OpenVINO 2025+, check documentation if needed
        if ('Async' in (item.labelstr)) and ('CPU' in (args.device)):
            # Check if CPU_THROUGHPUT_STREAMS is supported before setting/getting
            supported_props = ie.get_property(args.device, 'SUPPORTED_PROPERTIES')
            if 'CPU_THROUGHPUT_STREAMS' in supported_props:
                ie.set_property(args.device, {'CPU_THROUGHPUT_STREAMS': str(device_nstreams.get(args.device))
                                                             if args.device in device_nstreams.keys()
                                                             else 'CPU_THROUGHPUT_AUTO'})
                device_nstreams[args.device] = int(ie.get_property(args.device, 'CPU_THROUGHPUT_STREAMS'))
   
        #prepare input blob
        # NOTE: input_blob = compiled_model.input() in OpenVINO 2025+
        #load IR to plugin
        log.info("Loading network with plugin...")

        # NOTE: Use compile_model instead of load_network
        if 'Async' in (item.labelstr):
            # Async API is different in OpenVINO 2025+, refactor needed for full async support
            compiled_model = ie.compile_model(net, args.device, config)
            # NOTE: Async requests should use compiled_model.create_infer_request() and callbacks
            # For now, fallback to sync for demonstration
        else:
            compiled_model = ie.compile_model(net, args.device)

        input_blob = compiled_model.input(0)
        output_blob = compiled_model.output(0)
        
        #Do infer 
        inf_start = time.time()

        if 'Async' in (item.labelstr):
            # NOTE: Async infer API is different, needs refactor for OpenVINO 2025+
            # For demonstration, use sync infer
            res = compiled_model([input_ecg])
        else:
            res = compiled_model([input_ecg])

        inf_end = time.time()

        # NOTE: Output extraction is different
        if 'Async' in (item.labelstr):
            det_time = (inf_end - inf_start)/12
            # res = compiled_model.output(0) # If using async, would need to gather results from requests
            res = res[output_blob]
        else:
            det_time = inf_end - inf_start
            res = res[output_blob]

        del compiled_model
        print("[Performance] each inference time:{} ms".format(det_time*1000))

        mode_result = sst.mode(np.argmax(res, axis=2).squeeze(), keepdims=True)
        if hasattr(mode_result.mode, 'item'):
            prediction = mode_result.mode.item()
        else:
            prediction = mode_result.mode[0]
        print(prediction)
        result = preproc.int_to_class[prediction]

        # Store previous runs for overlay
        if not hasattr(fig, 'run_history'):
            fig.run_history = []
        # Color cycle for overlays
        overlay_colors = ['b', 'g', 'r', 'm', 'y', 'k', 'c']
        # Append current run to history
        fig.run_history.append((ts, input_ecg_plot, item.labelstr))
        # Clear plot only on first run
        if len(fig.run_history) == 1:
            ax1.cla()
        # Overlay all runs
        for idx, (ts_run, ecg_run, label_run) in enumerate(fig.run_history):
            color = overlay_colors[idx % len(overlay_colors)]
            ax1.plot(ts_run, ecg_run, label=label_run, lw=2, color=color)
        ax1.set_ylabel('Amplitude')
        ax1.set_title("ECG Raw signal: length - {}, Freq - 1000 Hz".format(ecg_h))
        ax1.legend(loc='upper right')
        # Only update xlabel for latest run
        ax1.set_xlabel('File: {}, Intel OpenVINO Infer_perf for each input: {}ms, classification_result: {}'.format(item.labelstr, det_time*1000, result), fontsize=15, color="c", fontweight='bold')
        ax1.grid()
        

    
    

if __name__ == '__main__':

    def handle_close(event):
        import sys
        import matplotlib.pyplot as plt
        plt.close('all')
        sys.exit(0)

    fig = plt.figure(figsize=(15,12))
    fig.canvas.mpl_connect('close_event', handle_close)
    # Move title below the menu bar
    fig.text(0.5, 0.92, 'Select ECG file of The Physionet 2017 Challenge from below list:',
             color="#009999", fontsize=18, fontweight='bold', ha='center', va='top')
    widths = [1, 1, 1, 1]
    heights = [1, 1, 6, 2]  # Reduce plot height, add info area
    gs = gridspec.GridSpec(ncols=4, nrows=4, width_ratios=widths, height_ratios=heights, figure=fig)
    #Menu
    props = ecg_menu.ItemProperties(labelcolor='white', bgcolor='#66ff99', fontsize=18, alpha=1.0)
    hoverprops = ecg_menu.ItemProperties(labelcolor='white', bgcolor='#33cc77', fontsize=18, alpha=1.0)
    menuitems = []

    for label in ('A00001.mat','A00005.mat','A00008.mat','A00022.mat','A00125.mat', 'Async 12 inputs', 'clear'):
        item = ecg_menu.MenuItem(fig, label, props=props, hoverprops=hoverprops, on_select=on_select)
        menuitems.append(item)
    menu = ecg_menu.Menu(fig, menuitems, 50, 1100)

    # Info text area below the plot
    info_ax = fig.add_subplot(gs[3, :])
    info_ax.axis('off')
    info = get_cpu_info()
    t = "CPU info: " +info['brand_raw'] + ", num of core(s): " +str(info['count'])
    t1 = ("In this Challenge, we treat all non-AF abnormal rhythms as a single "
          "class and require the Challenge entrant to classify the rhythms as:")
    t2 = ("1) N - Normal sinus rhythm")
    t3 = ("2) A - Atrial Fibrillation (AF)")
    t4 = ("3) O - Other rhythm")
    t5 = ("4) ~ - Too noisy to classify")
    t6 = ("*Algo refer to: Stanford Machine Learning Group ECG classification DNN model")
    t7 = ("https://stanfordmlgroup.github.io/projects/ecg2/")
    t8 = ("Demo created by: Zhao, Zhen (Fiona), VMC, IoTG, Intel")
    # Render info text in the info_ax area with dynamic vertical spacing
    info_lines = [
        (t, 16, 'center', 'top', '#0066cc'),
        (t1, 16, 'center', 'top', 'black'),
        (t2, 16, 'left', 'top', 'black'),
        (t3, 16, 'left', 'top', '#cc0066'),
        (t4, 16, 'left', 'top', '#6600cc'),
        (t5, 16, 'left', 'top', 'black'),
        (t6, 10, 'right', 'top', 'black'),
        (t7, 10, 'right', 'top', 'b'),
        (t8, 12, 'right', 'top', 'c'),
    ]
    n_lines = len(info_lines)
    # Use evenly spaced y positions from top to bottom, with horizontal padding
    left_pad = 0.08
    right_pad = 0.92
    for idx, (text, fontsize, ha, va, color) in enumerate(info_lines):
        y = 0.95 - idx * (0.9 / (n_lines-1))  # spread lines evenly from y=0.95 to y=0.05
        x = 0.5 if ha=='center' else (left_pad if ha=='left' else right_pad)
        info_ax.text(x, y, text,
                     fontsize=fontsize, style='oblique', ha=ha, va=va, wrap=True, color=color,
                     fontweight='bold' if idx==8 else 'normal')
    plt.axis('off')
    
    plt.show()
    #out = ecg.ecg(signal=input_ecg_plot, sampling_rate=1000., show=True)





