import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "duration": np.random.randint(0, 100, 500),
    "protocol_type": np.random.choice(["tcp", "udp", "icmp"], 500),
    "service": np.random.choice(["http", "ftp", "smtp", "dns"], 500),
    "src_bytes": np.random.randint(0, 10000, 500),
    "dst_bytes": np.random.randint(0, 10000, 500),
    "count": np.random.randint(0, 100, 500),
    "srv_count": np.random.randint(0, 100, 500),
    "same_srv_rate": np.random.rand(500),
    "diff_srv_rate": np.random.rand(500),
    "label": np.random.choice(["normal", "attack"], 500)
}

df = pd.DataFrame(data)
df.to_csv("C:\Users\lenovo\Desktop\Intrusion Detection System", index=False)

print("Custom dataset created!")
