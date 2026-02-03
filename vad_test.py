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

import nemo
import nemo.collections.asr as nemo_asr
vad_model = EncDecFrameClassificationModel.restore_from('models/vad_multilingual_marblenet.nemo', strict=False)
print("Load model successfull")
from omegaconf import OmegaConf
import copy
cfg = copy.deepcopy(vad_model._cfg)
print(OmegaConf.to_yaml(cfg))
vad_model.preprocessor = vad_model.from_config_dict(cfg.preprocessor)
vad_model.eval();
vad_model = vad_model.to(vad_model.device)

from nemo.core.classes import IterableDataset
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
import torch
from torch.utils.data import DataLoader

class AudioDataLayer(IterableDataset):
    @property
    def output_types(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)
        
    def set_signal(self, signal):
        self.signal = signal.astype(np.float32)/32768.
        self.signal_shape = self.signal.size
        self.output = True

    def __len__(self):
        return 1

data_layer = AudioDataLayer(sample_rate=cfg.train_ds.sample_rate)
data_loader = DataLoader(data_layer, batch_size=1, collate_fn=data_layer.collate_fn)

def infer_signal(model, signal):
    data_layer.set_signal(signal)
    batch = next(iter(data_loader))
    audio_signal, audio_signal_len = batch
    audio_signal, audio_signal_len = audio_signal.to(vad_model.device), audio_signal_len.to(vad_model.device)
    logits = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
    return logits

class FrameVAD:
    
    def __init__(self, model_definition,
                 threshold=0.5,
                 frame_len=2, frame_overlap=2.5, 
                 offset=10):
        '''
        Args:
          threshold: If prob of speech is larger than threshold, classify the segment to be speech.
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.vocab = list(model_definition['labels'])
        self.vocab.append('_')
        
        self.sr = model_definition['sample_rate']
        self.threshold = threshold
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = model_definition['AudioToMFCCPreprocessor']['window_stride']
        for block in model_definition['JasperEncoder']['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.reset()
        
    def _decode(self, frame, offset=0):
        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        logits = infer_signal(vad_model, self.buffer).cpu().numpy()[0]
        decoded = self._greedy_decoder(
            self.threshold,
            logits,
            self.vocab
        )
        return decoded  
    
    
    @torch.no_grad()
    def transcribe(self, frame=None):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged = self._decode(frame, self.offset)
        return unmerged
    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self.prev_char = ''

    @staticmethod
    def _greedy_decoder(threshold, logits, vocab):
        s = []
        if logits.shape[0]:
            probs = torch.softmax(torch.as_tensor(logits), dim=-1)
            probas, _ = torch.max(probs, dim=-1)
            probas_s = probs[1].item()
            preds = 1 if probas_s >= threshold else 0
            s = [preds, str(vocab[preds]), probs[0].item(), probs[1].item(), str(logits)]
        return s
    
STEP_LIST =        [0.01,0.01]
WINDOW_SIZE_LIST = [0.31,0.15]


import wave

def offline_inference(wave_file, STEP = 0.025, WINDOW_SIZE = 0.5, threshold=0.5):
    
    FRAME_LEN = STEP # infer every STEP seconds 
    CHANNELS = 1 # number of audio channels (expect mono signal)
    RATE = 16000 # sample rate, Hz
    
   
    CHUNK_SIZE = int(FRAME_LEN*RATE)
    
    vad = FrameVAD(model_definition = {
                   'sample_rate': SAMPLE_RATE,
                   'AudioToMFCCPreprocessor': cfg.preprocessor,
                   'JasperEncoder': cfg.encoder,
                   'labels': cfg.labels
               },
               threshold=threshold,
               frame_len=FRAME_LEN, frame_overlap = (WINDOW_SIZE-FRAME_LEN)/2,
               offset=0)

    wf = wave.open(wave_file, 'rb')
    p = pa.PyAudio()

    empty_counter = 0

    preds = []
    proba_b = []
    proba_s = []
    
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=CHANNELS,
                    rate=RATE,
                    output = True)

    data = wf.readframes(CHUNK_SIZE)

    while len(data) > 0:

        data = wf.readframes(CHUNK_SIZE)
        signal = np.frombuffer(data, dtype=np.int16)
        result = vad.transcribe(signal)

        preds.append(result[0])
        proba_b.append(result[2])
        proba_s.append(result[3])
        
        if len(result):
            print(result,end='\n')
            empty_counter = 3
        elif empty_counter > 0:
            empty_counter -= 1
            if empty_counter == 0:
                print(' ',end='')
                
    p.terminate()
    vad.reset()
    
    return preds, proba_b, proba_s
# import wave
# import numpy as np

# def offline_inference(wave_file, STEP=0.025, WINDOW_SIZE=0.5, threshold=0.5):

#     FRAME_LEN = STEP
#     CHANNELS = 1
#     RATE = 16000

#     CHUNK_SIZE = int(FRAME_LEN * RATE)

#     vad = FrameVAD(
#         model_definition={
#             'sample_rate': SAMPLE_RATE,
#             'AudioToMFCCPreprocessor': cfg.preprocessor,
#             'JasperEncoder': cfg.encoder,
#             'labels': cfg.labels
#         },
#         threshold=threshold,
#         frame_len=FRAME_LEN,
#         frame_overlap=(WINDOW_SIZE - FRAME_LEN) / 2,
#         offset=0
#     )

#     wf = wave.open(wave_file, 'rb')

#     preds = []
#     proba_b = []
#     proba_s = []

#     empty_counter = 0

#     while True:
#         data = wf.readframes(CHUNK_SIZE)
#         if not data:
#             break

#         signal = np.frombuffer(data, dtype=np.int16)
#         result = vad.transcribe(signal)

#         preds.append(result[0])
#         proba_b.append(result[2])
#         proba_s.append(result[3])

#         if len(result):
#             print(result)
#             empty_counter = 3
#         elif empty_counter > 0:
#             empty_counter -= 1
#             if empty_counter == 0:
#                 print(' ')

#     wf.close()
#     vad.reset()

#     return preds, proba_b, proba_s

STEP = 0.04
WINDOW_SIZE = 0.31
CHANNELS = 1 
RATE = 16000
FRAME_LEN = STEP
THRESHOLD = 0.95
SAMPLE_RATE = RATE
CHUNK_SIZE = int(STEP * RATE)
vad = FrameVAD(model_definition = {
                   'sample_rate': SAMPLE_RATE,
                   'AudioToMFCCPreprocessor': cfg.preprocessor,
                   'JasperEncoder': cfg.encoder,
                   'labels': cfg.labels
               },
               threshold=THRESHOLD,
               frame_len=FRAME_LEN, frame_overlap=(WINDOW_SIZE - FRAME_LEN) / 2, 
               offset=0)

import logging
import sys

# Cấu hình logging cơ bản
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Đảm bảo log ra console ngay lập tức
    ]
)
logger = logging.getLogger("VAD_App")

import time
import numpy as np
import pyaudio as pa

# Reset VAD (giả sử đối tượng vad đã tồn tại)
vad.reset()

import queue
import threading

#
# asr_queue = queue.Queue()

# # 2. Hàm Worker chạy ở luồng riêng
# def asr_worker():
#     while True:
#         # Lấy dữ liệu từ queue (sẽ đợi cho đến khi có dữ liệu)
#         audio_data = asr_queue.get()
#         if audio_data is None: break # Lệnh dừng
        
#         logger.info("ASR đang xử lý...")
#         # result = asr_model.transcribe(audio_data)
#         # logger.info(f"Kết quả ASR: {result}")
        
#         asr_queue.task_done()

# worker_thread = threading.Thread(target=asr_worker, daemon=True)
# worker_thread.start()

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