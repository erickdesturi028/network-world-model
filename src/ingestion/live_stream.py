import time
from collections import deque
import numpy as np
import torch
from scapy.all import IP, TCP, UDP, Ether

class LiveStreamBuffer:
    def __init__(self, window_packet_count=1, seq_len=10):
        self.window_packet_count = window_packet_count
        self.seq_len = seq_len
        self.packet_buffer = []
        self.state_sequence = deque(maxlen=seq_len)
        
    def extract_packet_features(self, pkt):
        """Extracts metrics from any arriving frame to avoid packet drops."""
        timestamp = pkt.time if hasattr(pkt, 'time') else time.time()
        
        # Check protocol or default to 0
        protocol = pkt[IP].proto if pkt.haslayer(IP) else 0
        
        syn = 1 if pkt.haslayer(TCP) and (pkt[TCP].flags & 0x02) else 0
        ack = 1 if pkt.haslayer(TCP) and (pkt[TCP].flags & 0x10) else 0
        fin = 1 if pkt.haslayer(TCP) and (pkt[TCP].flags & 0x01) else 0
        
        return {
            "time": timestamp,
            "protocol": protocol,
            "syn": syn,
            "ack": ack,
            "fin": fin
        }

    def process_packet(self, pkt):
        feat = self.extract_packet_features(pkt)
        self.packet_buffer.append(feat)

        # Build state vector immediately on every packet
        if len(self.packet_buffer) >= self.window_packet_count:
            times = [p["time"] for p in self.packet_buffer]
            iats = np.diff(times) if len(times) > 1 else [0.01]
            
            flow_iat_mean = float(np.mean(iats)) if len(iats) > 0 else 0.01
            flow_iat_std = float(np.std(iats)) if len(iats) > 0 else 0.00
            
            syn_cnt = sum(p["syn"] for p in self.packet_buffer)
            ack_cnt = sum(p["ack"] for p in self.packet_buffer)
            fin_cnt = sum(p["fin"] for p in self.packet_buffer)
            proto_mean = float(np.mean([p["protocol"] for p in self.packet_buffer]))
            
            syn_ack_ratio = float(syn_cnt / (ack_cnt + 1e-5))

            # 7-feature state vector (S_t)
            state_vector = np.array([
                proto_mean,
                flow_iat_mean,
                flow_iat_std,
                syn_cnt,
                ack_cnt,
                fin_cnt,
                syn_ack_ratio
            ], dtype=np.float32)

            # Basic scaling normalization
            norm_vector = np.clip(state_vector / (np.max(state_vector) + 1e-5), 0, 1)

            # Push directly into sliding sequence window
            self.state_sequence.append(norm_vector)
            self.packet_buffer = []

        # Return full [1, 10, 7] tensor once 10 steps exist
        if len(self.state_sequence) == self.seq_len:
            tensor_seq = torch.tensor(np.array(self.state_sequence), dtype=torch.float32).unsqueeze(0)
            return tensor_seq
        
        return None