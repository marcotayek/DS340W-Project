import urllib.request,json,gzip,csv,concurrent.futures,pathlib,collections,datetime
out=(pathlib.Path(__file__).resolve().parent / 'data'); out.mkdir(exist_ok=True)
expected={'EPL':380,'La_Liga':380,'Bundesliga':306,'Serie_A':380,'Ligue_1':306}
def fetch(league):
 url=f'https://understat.com/getLeagueData/{league}/2025'
 req=urllib.request.Request(url,headers={'X-Requested-With':'XMLHttpRequest','Referer':f'https://understat.com/league/{league}/2025'})
 with urllib.request.urlopen(req,timeout=60) as r: b=r.read()
 if b[:2]==b'\x1f\x8b': b=gzip.decompress(b)
 d=json.loads(b); matches=d['dates']; done=[m for m in matches if m['isResult']]
 assert len(done)==expected[league] and len(matches)==len(done),(league,len(done),len(matches))
 counts=collections.Counter()
 for m in done: counts[m['h']['id']]+=1; counts[m['a']['id']]+=1
 assert set(counts.values())==({34} if league in ['Bundesliga','Ligue_1'] else {38})
 (out/f'{league}_2025_26_source.json').write_text(json.dumps(d))
 rows=d['players']
 for row in rows:
  row.update(league=league,year=2025,season='2025/26',season_start=2025,is_partial_season=False,primary_position=row['position'].split()[0],source_url=url,retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 print(league,len(rows),len(done),'complete',flush=True)
 return rows,{'league':league,'player_rows':len(rows),'completed_matches':len(done),'expected_matches':expected[league],'teams':len(counts),'first_match':min(m['datetime'] for m in done),'last_match':max(m['datetime'] for m in done),'source_url':url}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex: results=list(ex.map(fetch,expected))
rows=[r for rs,_ in results for r in rs]
with (out/'player_stats_2025_26.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(out/'coverage_2025_26.json').write_text(json.dumps([s for _,s in results],indent=2))
print('TOTAL',len(rows))
