import sys,importlib.util,numpy as np,os
sys.path.insert(0,'/home/raemox/supertonic/.venv/lib/python3.10/site-packages')
from supertonic import TTS
tts=TTS(auto_download=False); sdk=tts.model.text_processor
spec=importlib.util.spec_from_file_location('bd','/home/raemox/stwork/board_driver.py')
# board_driver imports numpy/wave only at top; safe to load
bd=importlib.util.module_from_spec(spec)
# avoid running __main__
spec.loader.exec_module(bd)
tp=bd.TextProc(os.path.expanduser('~/.cache/supertonic3/onnx/unicode_indexer.json'))
lines=['Hi, I am Jero! Lets play.','Hello there friend.','What is your name?','Run fast now.','Give me five.','I love robots.','See you tomorrow.','That was amazing!']
allok=True
for ln in lines:
    sids,smask=sdk([ln],'en'); sids=sids[0]
    bids,T=tp.ids(ln,'en')
    ok=(len(sids)==len(bids)) and bool(np.all(sids==bids))
    allok&=ok
    if not ok: print('MISMATCH',repr(ln),'sdk',sids[:12],'board',bids[:12])
print('text_ids identical for all %d lines: %s'%(len(lines),allok))
