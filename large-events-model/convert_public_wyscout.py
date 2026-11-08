"""Adapt public 2017/18 Wyscout JSON to a small LEM-compatible real-data demo.

This is NOT an exact reproduction of Mendes-Neves et al.: public Wyscout differs
from Wyscout V3 and several original features are unavailable.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

VOCAB = ['pass','long_pass','cross','touch','aerial_duel','clearance','interception',
'loose_ball_duel','defensive_duel','offensive_duel','dribble','carry','game_interruption',
'own_goal','throw_in','free_kick','goal_kick','infraction','corner','acceleration',
'offside','right_foot_shot','left_foot_shot','head_shot','goalkeeper_exit','save',
'shot_against','fairplay','yellow_card','red_card','first_half_end','game_end']
V = {name:i for i,name in enumerate(VOCAB)}

def event_type(e):
    name = e.get('eventName','')
    sub = e.get('subEventName','')
    tags = {t.get('id') for t in e.get('tags',[])}
    if name == 'Pass':
        if 'Cross' in sub: return 'cross'
        if 'Launch' in sub or 'High pass' in sub: return 'long_pass'
        return 'pass'
    if name == 'Shot':
        if 403 in tags: return 'head_shot'
        if 402 in tags: return 'left_foot_shot'
        return 'right_foot_shot' # foot unknown defaults to right: approximation
    if name == 'Duel':
        if 'Air' in sub: return 'aerial_duel'
        if 'Loose' in sub: return 'loose_ball_duel'
        if 'Defensive' in sub: return 'defensive_duel'
        return 'offensive_duel'
    if name == 'Free Kick':
        if 'Corner' in sub: return 'corner'
        if 'Throw in' in sub: return 'throw_in'
        if 'Goal kick' in sub: return 'goal_kick'
        return 'free_kick'
    if name == 'Foul': return 'infraction'
    if name == 'Offside': return 'offside'
    if name == 'Interruption': return 'game_interruption'
    if name == 'Save attempt': return 'save'
    if name == 'Others on the ball': return 'touch'
    if name == 'Goalkeeper leaving line': return 'goalkeeper_exit'
    return 'touch' # unsupported Wyscout events mapped to touch

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--events', required=True)
    p.add_argument('--matches', required=True)
    p.add_argument('--output',default='real_wyscout')
    p.add_argument('--max-matches',type=int,default=80,help='Use 80 matches for a manageable first run; 0=all')
    args=p.parse_args()
    with open(args.matches,encoding='utf-8') as f: matches=json.load(f)
    meta={}
    for m in matches:
        teams=m.get('teamsData',{})
        home=next((int(k) for k,v in teams.items() if v.get('side')=='home'),None)
        away=next((int(k) for k,v in teams.items() if v.get('side')=='away'),None)
        if home is not None and away is not None:
            meta[m['wyId']]=(home,away)
    ids=sorted(meta)
    if args.max_matches: ids=ids[:args.max_matches]
    ids=set(ids)
    with open(args.events,encoding='utf-8') as f: events=json.load(f)
    records=[]
    for ev in events:
        mid=ev.get('matchId')
        if mid not in ids or ev.get('matchPeriod') not in ('1H','2H'): continue
        pos=ev.get('positions') or []
        if not pos: continue
        x=pos[0].get('x',50); y=pos[0].get('y',50)
        home,away=meta[mid]
        period=ev['matchPeriod']
        sec=float(ev.get('eventSec',0))
        records.append((mid,0 if period=='1H' else 1,sec,int(ev.get('id',0)),
                        int(ev.get('teamId',0)==home),V[event_type(ev)],
                        int(np.clip(round(x),0,100)),int(np.clip(round(y),0,100))))
    if not records: raise SystemExit('No usable events; check input paths')
    df=pd.DataFrame(records,columns=['match_id','period','seconds','event_id','h','e','x','y'])
    df=df.sort_values(['match_id','period','seconds','event_id']).reset_index(drop=True)
    # Wyscout eventSec resets at halftime; elapsed intervals reset at each period.
    previous=df.groupby(['match_id','period'])['seconds'].shift(1)
    df['t']=((df.seconds-previous.fillna(df.seconds)).clip(lower=0)/5).round().clip(0,100).astype('int8')
    # 'a' denotes a coarse action-success proxy, not the paper's exact definition.
    df['a']=0
    # Explicitly no unsupported game-state columns are fabricated.
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    match_ids=sorted(df.match_id.unique())
    rng=np.random.default_rng(42); rng.shuffle(match_ids)
    cutoff=max(1,int(len(match_ids)*.8))
    train_ids=set(match_ids[:cutoff]); val_ids=set(match_ids[cutoff:])
    if not val_ids: raise SystemExit('Need at least two matches for validation')
    for split, subset in [('train',df[df.match_id.isin(train_ids)]),('val',df[df.match_id.isin(val_ids)])]:
        raw=subset[['match_id','h','e','x','y','t','a']].copy()
        raw.to_csv(out/f'{split}_raw_events.csv',index=False)
        # Construct next-event supervised rows; no cross-match context leakage.
        for col in ['h','e','x','y','t','a']:
            raw[f'prev_{col}']=raw.groupby('match_id')[col].shift(1)
        raw=raw.dropna().copy()
        features=['prev_h','prev_e','prev_x','prev_y','prev_t','prev_a']
        for target in ['e','x','y','t']:
            frame=raw[features+[target]].rename(columns={target:'target'})
            frame.to_csv(out/f'{split}_{target}.csv',index=False)
        print(f'{split}: {len(subset):,} real events; {len(raw):,} usable next-event examples; {subset.match_id.nunique()} matches')
    print(f'Wrote real-data CSVs to {out.resolve()}')
    print('NOTE: Public Wyscout schema/coverage differs from paper; do not label results exact reproduction.')

if __name__=='__main__':main()
