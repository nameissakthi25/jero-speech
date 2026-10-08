import os,glob,json,numpy as np,onnx,onnxruntime as ort
from aimet_onnx.quantsim import QuantizationSimModel
from aimet_onnx.common.defs import qtype, QuantScheme
import aimet_onnx
cfg=os.path.join(os.path.dirname(aimet_onnx.__file__),"common","quantsim_config","htp_quantsim_config_v68.json")
enc_onnx="encsplit/intent_encoder_m50.onnx"
fp=ort.InferenceSession(enc_onnx,providers=["CPUExecutionProvider"])
ref=json.load(open("encsplit/verify/fp32_ref.json"))
# calib inputs
calib=[]
for i in range(100):
    ie=np.fromfile("encsplit/calib/ie_%03d.raw"%i,dtype=np.float32).reshape(1,32,768)
    m=np.fromfile("encsplit/calib/mask_%03d.raw"%i,dtype=np.int32).reshape(1,32).astype(np.int64)
    calib.append({"inputs_embeds":ie,"attention_mask":m})
sim=QuantizationSimModel(model=onnx.load(enc_onnx),param_type=qtype.int(8),activation_type=qtype.int(16),
  quant_scheme=QuantScheme.min_max,config_file=cfg,dummy_input=calib[0])
sim.compute_encodings(iter(calib))
cs=[]
for i in range(10):
    ie=np.fromfile("encsplit/verify/ie_%03d.raw"%i,dtype=np.float32).reshape(1,32,768)
    m=np.fromfile("encsplit/verify/mask_%03d.raw"%i,dtype=np.int32).reshape(1,32).astype(np.int64)
    a=fp.run(None,{"inputs_embeds":ie,"attention_mask":m})[0]
    b=sim.session.run(None,{"inputs_embeds":ie,"attention_mask":m})[0]
    n=ref[i]["nreal"]
    x=a.reshape(32,768)[:n].ravel(); y=b.reshape(32,768)[:n].ravel()
    cs.append(float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-9)))
print("AIMET w8a16 host-sim cosine (encoder) mean=%.4f"%np.mean(cs), "per:",[round(c,3) for c in cs])
