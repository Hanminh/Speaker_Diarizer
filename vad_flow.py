import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.classification_models import EncDecFrameClassificationModel, EncDecClassificationModel
import numpy as np 
import librosa
import soundfile as sf 
import torch 
import inspect
import numpy as np
import pyaudio as pa
import os, time
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
# %matplotlib inline
from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
from torch.utils.data import DataLoader
from vad_model import FrameVAD
import time
import numpy as np
import pyaudio as pa
import queue
import threading
import logging
import sys

STEP = 0.04
WINDOW_SIZE = 0.31
CHANNELS = 1 
RATE = 16000
FRAME_LEN = STEP
THRESHOLD = 0.95
SAMPLE_RATE = 16000
CHUNK_SIZE = int(STEP * RATE)

# Cấu hình logging cơ bản
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Đảm bảo log ra console ngay lập tức
    ]
)
logger = logging.getLogger("VAD_App")

vad_model = EncDecFrameClassificationModel.restore_from('models/vad_multilingual_marblenet.nemo', strict=False)
print("Load model successfull")

from omegaconf import OmegaConf
import copy
cfg = copy.deepcopy(vad_model._cfg)
print(OmegaConf.to_yaml(cfg))

vad_model.preprocessor = vad_model.from_config_dict(cfg.preprocessor)
vad_model.eval();
vad_model = vad_model.to(vad_model.device)

vad = FrameVAD(model_definition = {
                   'sample_rate': SAMPLE_RATE,
                   'AudioToMFCCPreprocessor': cfg.preprocessor,
                   'JasperEncoder': cfg.encoder,
                   'labels': cfg.labels
               },
               threshold=THRESHOLD,
               frame_len=FRAME_LEN, frame_overlap=(WINDOW_SIZE - FRAME_LEN) / 2, 
               offset=0)

p = pa.PyAudio()
logger.info('Searching for audio input devices...')

input_devices = []
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('maxInputChannels'):
        input_devices.append(i)
        # Thay print bằng logger.info
        logger.info(f"ID {i}: {dev.get('name')}")

if len(input_devices):
    dev_idx = -2
    while dev_idx not in input_devices:
        try:
            val = input('Please type input device ID: ')
            dev_idx = int(val)
        except ValueError:
            logger.error("Invalid input. Please enter a number.")

    empty_counter = 0

    def callback(in_data, frame_count, time_info, status):
        global empty_counter

        signal = np.frombuffer(in_data, dtype=np.int16)
        text = vad.transcribe(signal)
        
        if len(text):
            if text[0] == 1:
            # if True: 
                logger.info(f"Result: {text}")
                logger.info(f'Offset: {vad.offset}')
            empty_counter = vad.offset
        elif empty_counter > 0:
            empty_counter -= 1
            if empty_counter == 0:
                logger.debug("Silence detected.") 
                
        return (in_data, pa.paContinue)

    try:
        stream = p.open(format=pa.paInt16,
                        channels=CHANNELS,
                        rate=SAMPLE_RATE,
                        input=True,
                        input_device_index=dev_idx,
                        stream_callback=callback,
                        frames_per_buffer=CHUNK_SIZE)

        logger.info('Microphone is now Listening... (Press Ctrl+C to stop)')
        stream.start_stream()
        
        while stream.is_active():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    finally:        
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()
        logger.info("PyAudio stopped and resources cleaned up.")
    
else:
    logger.critical('No audio input device found. Check WSLg/USBIPD connection.')

