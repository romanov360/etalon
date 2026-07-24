import json

raw = open('/private/tmp/claude-501/-Users-tr-prog-silicon-photonics/0f249ccb-e47a-434e-aaa3-0e68e05f220e/tasks/wkpde1vcl.output').read()
start = raw.find('{')
data = json.loads(raw[start:])
if isinstance(data, dict) and isinstance(data.get('result'), str):
    data = json.loads(data['result'])

theses = data.get('ranked_theses', [])
print("N theses:", len(theses))
for t in theses:
    th = t['thesis']
    print(f"\n{'='*80}\n{t['avg_score']}/10 — {th['name']}")
    print("ONE-LINER:", th['one_liner'])
    print("WEDGE:", th['wedge'][:600])
    print("CAPITAL:", th.get('capital_intensity', '')[:300])
    print("SOFTWARE MVP:", th.get('software_mvp', '')[:500])
    for v in t.get('verdicts', []):
        print(f"  - judge {v['score']}/10: {v['rationale'][:220]}")
        if v.get('fatal_flaws'):
            print(f"    flaws: {'; '.join(f[:120] for f in v['fatal_flaws'][:3])}")

# also dump full JSON to a workable location
with open('/tmp/research_result.json', 'w') as f:
    json.dump(data, f, indent=1)
print("\nsaved /tmp/research_result.json")
