import sqlite3
import json
import base64
import numpy as np
from PIL import Image
import io

from spectral_cd import run_spectral_inference

conn = sqlite3.connect('storage/geonexus_predictions.db')
c = conn.cursor()
c.execute("SELECT result_json FROM predictions WHERE id='spec_9f458da7'")
row = c.fetchone()
if row:
    res = json.loads(row[0])
    
    data = np.load("scratch_tensors.npz")
    t1_raw = data["t1"]
    t2_raw = data["t2"]
    
    t2_b64 = res["t2_rgb"].replace("data:image/png;base64,", "")
    
    updated_result = run_spectral_inference(
        t1_tensor=t1_raw,
        t2_tensor=t2_raw,
        t2_rgb_b64=t2_b64
    )
    
    res["change_mask"] = f"data:image/png;base64,{updated_result['change_mask_b64']}"
    res["binary_mask"] = f"data:image/png;base64,{updated_result['binary_mask_b64']}"
    res["change_prob"] = f"data:image/png;base64,{updated_result['change_prob_b64']}"
    res["overlay"]     = f"data:image/png;base64,{updated_result['overlay_b64']}"
    res["dynamics"]    = updated_result["dynamics"]
    res["stats"]       = updated_result["stats"]
    
    c.execute(
        "UPDATE predictions SET dynamics_json = ?, stats_json = ?, result_json = ? WHERE id='spec_9f458da7'",
        (json.dumps(updated_result["dynamics"]), json.dumps(updated_result["stats"]), json.dumps(res))
    )
    conn.commit()
    print("Updated spec_9f458da7 in SQLite database successfully!")
conn.close()
