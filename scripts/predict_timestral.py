"""Pinned Timestral inference worker. Input contains histories only, no labels."""
from pathlib import Path
import sys, argparse, time
import numpy as np
import torch
from timestral import TimestralForecaster
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from volrisk.io import write_json,sha256,utc_now

def main():
    p=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(2026)
    model=TimestralForecaster.from_pretrained(str(Path(__file__).resolve().parents[1]/'data/models/timestral-5m-bdd20cf'))
    contexts=np.load(a.input,allow_pickle=False); predictions=[];floored=0;t=time.monotonic()
    for start in range(0,len(contexts),64):
        result=model.forecast(contexts[start:start+64],prediction_length=21)
        quantiles=np.asarray(result.quantiles)[:,0,20,:]
        levels=np.asarray(result.quantile_levels)
        floored+=int((quantiles<1e-6).sum());quantiles=np.maximum(quantiles,1e-6)
        q=np.concatenate([quantiles[:,:1],quantiles,quantiles[:,-1:]],axis=1)
        u=np.r_[0.,levels,1.]
        predictions.extend(np.sqrt(np.trapezoid(q*q,u,axis=1)).tolist())
        if start%4096==0:print(start,len(contexts),flush=True)
    np.save(a.output,np.asarray(predictions))
    write_json(a.output+'.json',{'created_utc':utc_now(),'n':len(contexts),'floored_quantiles':floored,'seconds':time.monotonic()-t,'input_sha256':sha256(a.input),'output_sha256':sha256(a.output)})
if __name__=='__main__':main()
